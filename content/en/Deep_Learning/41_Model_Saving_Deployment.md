[Previous: Practical Text Classification Project](./40_Practical_Text_Classification.md) | [Next: Reinforcement Learning Introduction](./42_Reinforcement_Learning_Intro.md)

---

# 41. Model Saving and Deployment

## Learning Objectives

After completing this lesson, you will be able to:

1. Save and load PyTorch models using state_dict and checkpoints
2. Export models with TorchScript (tracing and scripting)
3. Convert models to ONNX for cross-framework deployment
4. Use `torch.export()` for graph capture (PyTorch 2.x)
5. Apply inference optimization techniques (quantization, compilation)
6. Deploy models via REST API, Docker, mobile, and cloud platforms

---

## 1. PyTorch Model Saving

### Saving state_dict (Recommended)

```python
# Save
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model = MyModel()  # Same structure required
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

### Saving Full Model

```python
# Save
torch.save(model, 'model_full.pth')

# Load
model = torch.load('model_full.pth')
model.eval()
```

### Saving Checkpoint

```python
# Save
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_acc': best_acc
}
torch.save(checkpoint, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 2. TorchScript

### Concept

```
Run model without Python dependency
- Loadable from C++
- Mobile deployment
- Server optimization
```

### Tracing

```python
# Trace with example input
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save('model_traced.pt')

# Load
loaded_model = torch.jit.load('model_traced.pt')
output = loaded_model(example_input)
```

### Scripting

```python
# Model with control flow
class MyModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        return x

scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

### Comparison

| Method | Advantages | Disadvantages |
|--------|-----------|---------------|
| Trace | Simple, works for most cases | Cannot handle dynamic control flow |
| Script | Supports dynamic control flow | Some Python features restricted |

---

## 3. ONNX Conversion

### Conversion

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11
)
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

# Create session
session = ort.InferenceSession("model.onnx")

# Inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: input_data})
```

### Validation

```python
import onnx

# Load and validate model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model validation passed")
```

---

## 4. torch.export() (PyTorch 2.x)

### Concept

`torch.export()` is the PyTorch 2.x successor to TorchScript. It captures the full computation graph using TorchDynamo, producing a clean, portable `ExportedProgram`.

```
TorchScript (legacy) → torch.export() (PyTorch 2.x recommended)
- Sound graph capture (no silent correctness issues)
- Works with torch.compile() ecosystem
- Better support for dynamic shapes
```

### Basic Usage

```python
import torch

model = MyModel().eval()
example_input = (torch.randn(1, 3, 224, 224),)

# Export the model
exported = torch.export.export(model, example_input)

# Run the exported model
output = exported.module()(torch.randn(1, 3, 224, 224))
```

### Dynamic Shapes

```python
from torch.export import Dim

# Define dynamic dimensions
batch = Dim("batch", min=1, max=32)
dynamic_shapes = {"x": {0: batch}}

exported = torch.export.export(
    model,
    (torch.randn(4, 3, 224, 224),),
    dynamic_shapes=dynamic_shapes,
)

# Works with any batch size in range [1, 32]
output = exported.module()(torch.randn(8, 3, 224, 224))
```

### Saving and Loading

```python
# Save
torch.export.save(exported, "model_exported.pt2")

# Load
loaded = torch.export.load("model_exported.pt2")
output = loaded.module()(input_data)
```

### torch.export() vs TorchScript

| Feature | TorchScript | torch.export() |
|---------|------------|----------------|
| Graph capture | Tracing or scripting | TorchDynamo (automatic) |
| Python support | Limited subset | Full Python semantics |
| Dynamic shapes | Manual annotation | First-class support |
| Correctness | Silent failures possible | Sound capture or clear error |
| Ecosystem | Legacy | Integrates with torch.compile |

---

## 5. Inference Optimization

### eval Mode

```python
model.eval()  # Disable Dropout and BatchNorm
```

### no_grad

```python
with torch.no_grad():
    output = model(input)
```

### Inference Mode (PyTorch 2.0+)

```python
with torch.inference_mode():
    output = model(input)
```

### Quantization

```python
# Dynamic quantization (simple)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Static quantization (more optimized)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)
# Run with calibration data
model_quantized = torch.quantization.convert(model_prepared)
```

---

## 6. Deployment Options

### Flask API

```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
model = torch.load('model.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    tensor = torch.tensor(data).float()

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).tolist()

    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI (Recommended)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = torch.jit.load('model_traced.pt')
model.eval()

class InputData(BaseModel):
    data: list

@app.post("/predict")
async def predict(input_data: InputData):
    tensor = torch.tensor(input_data.data).float()

    with torch.inference_mode():
        output = model(tensor)
        pred = output.argmax(dim=1).tolist()

    return {"prediction": pred}
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model_traced.pt .
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 7. Mobile Deployment

### PyTorch Mobile

```python
# Optimize for mobile
traced_model = torch.jit.trace(model, example_input)
optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
optimized_model._save_for_lite_interpreter("model_mobile.ptl")
```

### Android/iOS

```kotlin
// Android (Kotlin)
val module = LiteModuleLoader.load(assetFilePath(this, "model_mobile.ptl"))
val inputTensor = Tensor.fromBlob(inputArray, longArrayOf(1, 3, 224, 224))
val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
```

---

## 8. Cloud Deployment

### AWS SageMaker

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py'
)

predictor = model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)
```

### Hugging Face Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="model.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="username/model-name",
    repo_type="model"
)
```

---

## 9. Best Practices

### Pre-Save Checklist

```python
# 1. eval mode
model.eval()

# 2. GPU → CPU (portability)
model.cpu()

# 3. Validation
with torch.no_grad():
    test_output = model(test_input.cpu())
    assert test_output.shape == expected_shape
```

### Version Control

```python
save_dict = {
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_size': 784,
        'hidden_size': 256,
        'num_classes': 10
    },
    'pytorch_version': torch.__version__,
    'training_date': datetime.now().isoformat()
}
torch.save(save_dict, 'model_v1.0.pth')
```

---

## Summary

### Choosing Save Method

| Use Case | Method |
|----------|--------|
| Resume training | Checkpoint (state_dict + optimizer) |
| Python deployment | state_dict |
| C++ deployment | TorchScript |
| Universal deployment | ONNX |
| PyTorch 2.x ecosystem | torch.export() |
| Mobile | PyTorch Mobile |

### Core Code

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# TorchScript (legacy)
traced = torch.jit.trace(model.eval(), example_input)
traced.save('model.pt')

# torch.export() (PyTorch 2.x recommended)
exported = torch.export.export(model.eval(), (example_input,))
torch.export.save(exported, 'model.pt2')

# ONNX
torch.onnx.export(model, example_input, 'model.onnx')
```

---

## Next Steps

Proceed with practical projects in [39_Practical_Image_Classification.md](./39_Practical_Image_Classification.md).
