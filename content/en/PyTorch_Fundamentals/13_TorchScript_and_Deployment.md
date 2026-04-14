# TorchScript and Deployment

**Previous**: [Custom Layers and Functions](./12_Custom_Layers_and_Functions.md) | **Next**: [PyTorch Ecosystem](./14_PyTorch_Ecosystem.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between eager mode and TorchScript
2. Convert models to TorchScript using tracing and scripting
3. Understand when to use `torch.jit.trace` vs `torch.jit.script`
4. Use `torch.export` (PyTorch 2.x) for graph capture
5. Export models to ONNX for cross-platform deployment
6. Deploy PyTorch models via TorchServe
7. Optimize models for inference with quantization basics

---

Training is only half the story. Deploying models efficiently -- to servers, mobile devices, or edge hardware -- requires converting from eager Python to optimized representations.

---

## 1. TorchScript Overview

TorchScript is a way to serialize and optimize PyTorch models for deployment without requiring a Python runtime:

```
Eager Mode (Python)          TorchScript (Serialized)
┌──────────────────┐         ┌──────────────────────┐
│ model.forward()  │  ──▶    │ Intermediate         │
│ Python execution │         │ Representation (IR)  │
│ Full flexibility │         │ No Python needed     │
└──────────────────┘         │ Optimized execution  │
                             └──────────────────────┘
```

### 1.1 Two Conversion Methods

| Method | How | When |
|--------|-----|------|
| **Tracing** | Run model with sample input, record operations | Models without control flow (no if/for based on input) |
| **Scripting** | Parse Python source code into IR | Models with dynamic control flow |

---

## 2. Tracing

### 2.1 Basic Tracing

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

model = SimpleModel()
model.eval()

# Trace with example input
example_input = torch.randn(1, 784)
traced_model = torch.jit.trace(model, example_input)

# Use the traced model
output = traced_model(torch.randn(5, 784))
print(output.shape)  # [5, 10]

# Save
traced_model.save('traced_model.pt')

# Load (no Python model definition needed!)
loaded = torch.jit.load('traced_model.pt')
output = loaded(torch.randn(1, 784))
```

### 2.2 Tracing Limitations

```python
class ConditionalModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:       # data-dependent control flow
            return x * 2
        else:
            return x * 3

model = ConditionalModel()

# Tracing will only capture ONE branch!
traced = torch.jit.trace(model, torch.tensor([1.0]))  # captures x*2 branch
print(traced(torch.tensor([-1.0])))  # WRONG: still does x*2, not x*3
```

> **Rule**: If your model has `if`, `for`, or `while` statements that depend on input data, use scripting instead of tracing.

---

## 3. Scripting

### 3.1 Basic Scripting

```python
class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        if x.sum() > 0:
            return torch.relu(self.linear(x))
        else:
            return torch.sigmoid(self.linear(x))

model = DynamicModel()
model.eval()

# Script the model (parses Python source)
scripted_model = torch.jit.script(model)

# Both branches work correctly
print(scripted_model(torch.ones(1, 10)))    # uses relu
print(scripted_model(-torch.ones(1, 10)))   # uses sigmoid

scripted_model.save('scripted_model.pt')
```

### 3.2 Scripting Standalone Functions

```python
@torch.jit.script
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        0.7978845608 * (x + 0.044715 * x ** 3)
    ))

result = gelu(torch.randn(3))
```

### 3.3 Scripting Limitations

TorchScript supports a subset of Python. These are NOT supported:
- `**kwargs`
- Complex Python data structures (custom classes without `@torch.jit.script`)
- Many standard library functions
- Dynamic module creation

---

## 4. torch.export (PyTorch 2.x)

### 4.1 ExportedProgram

```python
# torch.export is the modern replacement for TorchScript
# It captures a clean computation graph while preserving semantics

model = SimpleModel()
model.eval()

example_input = (torch.randn(1, 784),)

# Export
exported = torch.export.export(model, example_input)
print(exported)

# Run the exported model
output = exported.module()(torch.randn(5, 784))

# Save and load
torch.export.save(exported, 'exported_model.pt2')
loaded = torch.export.load('exported_model.pt2')
```

### 4.2 Dynamic Shapes with torch.export

```python
from torch.export import Dim

batch = Dim("batch", min=1, max=256)
exported = torch.export.export(
    model,
    (torch.randn(1, 784),),
    dynamic_shapes={"x": {0: batch}},
)

# Works with any batch size in [1, 256]
output = exported.module()(torch.randn(32, 784))
```

---

## 5. torch.compile (Runtime Optimization)

### 5.1 Basic Usage

```python
model = SimpleModel().to(device)

# Compile for faster execution
compiled_model = torch.compile(model)

# First call: compiles (slow)
# Subsequent calls: optimized (fast)
output = compiled_model(torch.randn(32, 784, device=device))
```

### 5.2 Compile Modes

```python
# Default: good balance of compile time and speedup
model = torch.compile(model)

# Reduce-overhead: minimize CPU overhead (good for small models)
model = torch.compile(model, mode="reduce-overhead")

# Max-autotune: try all optimizations (slow compile, fastest runtime)
model = torch.compile(model, mode="max-autotune")
```

### 5.3 Compile vs TorchScript vs Export

| Feature | torch.compile | TorchScript | torch.export |
|---------|--------------|-------------|-------------|
| **Goal** | Runtime speed | Serialization | Graph capture |
| **Saves to disk** | No | Yes | Yes |
| **Python needed** | Yes | No | No |
| **Control flow** | Full Python | Subset | Guards |
| **Performance** | Best | Good | Good |
| **PyTorch version** | 2.0+ | 1.0+ | 2.1+ |

---

## 6. ONNX Export

### 6.1 Export with Dynamic Axes

```python
model = SimpleModel()
model.eval()

dummy_input = torch.randn(1, 784)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=17,
)
```

### 6.2 Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")

# CPU inference
input_data = np.random.randn(10, 784).astype(np.float32)
outputs = session.run(None, {"input": input_data})
print(f"Predictions shape: {outputs[0].shape}")  # (10, 10)

# GPU inference (if available)
# session = ort.InferenceSession("model.onnx",
#     providers=['CUDAExecutionProvider'])
```

---

## 7. TorchServe

### 7.1 Packaging a Model

```bash
# Install TorchServe
pip install torchserve torch-model-archiver

# Create a model archive
torch-model-archiver \
    --model-name my_model \
    --version 1.0 \
    --serialized-file model_weights.pt \
    --handler handler.py \
    --export-path model_store/

# Start TorchServe
torchserve --start --model-store model_store --models my_model=my_model.mar
```

### 7.2 Custom Handler

```python
# handler.py
import torch
import torch.nn as nn
from ts.torch_handler.base_handler import BaseHandler

class MyHandler(BaseHandler):
    def initialize(self, context):
        self.model = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)
        )
        state_dict = torch.load(context.model_dir + '/model_weights.pt',
                                 map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, data):
        return torch.tensor(data[0]['body'], dtype=torch.float32)

    def inference(self, input_data):
        with torch.no_grad():
            return self.model(input_data)

    def postprocess(self, output):
        return output.argmax(dim=1).tolist()
```

---

## 8. Quantization (Brief Introduction)

### 8.1 Dynamic Quantization

```python
# Quantize Linear and LSTM layers at runtime
quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {nn.Linear},          # layers to quantize
    dtype=torch.qint8     # quantize to int8
)

# Model size comparison
import os
torch.save(model.state_dict(), 'fp32_model.pt')
torch.save(quantized_model.state_dict(), 'int8_model.pt')

fp32_size = os.path.getsize('fp32_model.pt') / 1024
int8_size = os.path.getsize('int8_model.pt') / 1024
print(f"FP32: {fp32_size:.1f} KB")
print(f"INT8: {int8_size:.1f} KB")
print(f"Compression: {fp32_size/int8_size:.1f}x")
```

---

## Summary

| Tool | Purpose | Key Command |
|------|---------|-------------|
| Tracing | Serialize models without control flow | `torch.jit.trace(model, input)` |
| Scripting | Serialize models with control flow | `torch.jit.script(model)` |
| torch.export | Modern graph capture (PyTorch 2.x) | `torch.export.export(model, args)` |
| torch.compile | Runtime optimization (no serialization) | `torch.compile(model)` |
| ONNX | Cross-framework export | `torch.onnx.export(model, input, path)` |
| TorchServe | Model serving | `torch-model-archiver` + `torchserve` |
| Quantization | Model compression | `quantize_dynamic(model, ...)` |

---

**Next**: [PyTorch Ecosystem](./14_PyTorch_Ecosystem.md) -- Libraries and tools that extend PyTorch.
