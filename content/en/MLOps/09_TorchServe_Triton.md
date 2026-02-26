# 09. TorchServe & Triton Inference Server

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the TorchServe architecture including its frontend, backend, model store, and worker components, and explain how requests are processed
2. Package a PyTorch model into a `.mar` archive file and deploy it with TorchServe using custom request handlers
3. Explain the Triton Inference Server architecture and its support for multiple backends (PyTorch, TensorFlow, ONNX, TensorRT)
4. Configure Triton model repositories with ensemble pipelines and dynamic batching to maximize GPU throughput
5. Compare TorchServe and Triton Inference Server across performance, flexibility, and operational complexity to select the appropriate tool for a deployment scenario

---

## 1. TorchServe Overview

TorchServe is the official tool for serving PyTorch models in production environments.

### 1.1 TorchServe Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TorchServe Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐                                                   │
│   │   Client    │                                                   │
│   └──────┬──────┘                                                   │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                    Frontend                              │      │
│   │  ┌──────────────┐  ┌──────────────┐                     │      │
│   │  │ REST API     │  │ gRPC API     │                     │      │
│   │  │ :8080        │  │ :7070        │                     │      │
│   │  └──────────────┘  └──────────────┘                     │      │
│   └─────────────────────────────────────────────────────────┘      │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                    Backend                               │      │
│   │  ┌──────────────────────────────────────────┐           │      │
│   │  │              Model Store                  │           │      │
│   │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │           │      │
│   │  │  │ Model A │ │ Model B │ │ Model C │    │           │      │
│   │  │  └─────────┘ └─────────┘ └─────────┘    │           │      │
│   │  └──────────────────────────────────────────┘           │      │
│   │                                                          │      │
│   │  ┌──────────────┐  ┌──────────────┐                     │      │
│   │  │ Worker 1     │  │ Worker 2     │  ...                │      │
│   │  └──────────────┘  └──────────────┘                     │      │
│   └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Installation

```bash
# Install TorchServe
pip install torchserve torch-model-archiver torch-workflow-archiver

# Check version
torchserve --version
```

---

## 2. Writing Handlers

### 2.1 Basic Handler

```python
"""
custom_handler.py - TorchServe custom handler
"""

import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
import json
import logging

logger = logging.getLogger(__name__)

class CustomHandler(BaseHandler):
    """Custom TorchServe handler"""

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        """Model initialization"""
        # Called once at startup, not per-request — load model and allocate GPU memory here
        # to avoid 100-500ms overhead on every inference call
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Auto-detect GPU — falls back to CPU so the same handler works in both environments
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = f"{model_dir}/{serialized_file}"

        # TorchScript model — serialized computation graph that doesn't need Python at runtime,
        # enabling faster inference and C++ deployment
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.model.eval()

        # Load additional config (if exists)
        self.class_names = self._load_class_names(model_dir)

        self.initialized = True
        logger.info("Model initialized successfully")

    def _load_class_names(self, model_dir):
        """Load class names"""
        try:
            with open(f"{model_dir}/index_to_name.json") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def preprocess(self, data):
        """Input preprocessing"""
        inputs = []

        for row in data:
            # Process JSON input
            if isinstance(row, dict):
                features = row.get("data") or row.get("body")
            else:
                features = row.get("body")

            if isinstance(features, (bytes, bytearray)):
                features = json.loads(features.decode("utf-8"))

            tensor = torch.tensor(features, dtype=torch.float32)
            inputs.append(tensor)

        # Stack into batch
        return torch.stack(inputs).to(self.device)

    def inference(self, data):
        """Perform inference"""
        # torch.no_grad() disables gradient tracking — reduces memory usage by ~50%
        # and speeds up inference since backprop is never needed at serving time
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities

    def postprocess(self, data):
        """Output postprocessing"""
        results = []

        for prob in data:
            prob_list = prob.cpu().numpy().tolist()
            prediction = int(torch.argmax(prob).item())

            result = {
                "prediction": prediction,
                "probabilities": prob_list
            }

            # Add class name
            if self.class_names:
                result["class_name"] = self.class_names.get(str(prediction))

            results.append(result)

        return results
```

### 2.2 Image Classification Handler

```python
"""
image_classifier_handler.py - Image classification handler
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler
from PIL import Image
import io
import base64

class ImageClassifierHandler(VisionHandler):
    """Image classification handler"""

    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, data):
        """Image preprocessing"""
        images = []

        for row in data:
            image_data = row.get("data") or row.get("body")

            # Base64 decoding
            if isinstance(image_data, str):
                image_data = base64.b64decode(image_data)

            # Load image
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image = self.transform(image)
            images.append(image)

        return torch.stack(images).to(self.device)

    def postprocess(self, data):
        """Result postprocessing"""
        probabilities = F.softmax(data, dim=1)
        top_k = torch.topk(probabilities, 5)

        results = []
        for probs, indices in zip(top_k.values, top_k.indices):
            result = {
                "predictions": [
                    {
                        "class_id": int(idx),
                        "class_name": self.mapping.get(str(int(idx)), "unknown"),
                        "probability": float(prob)
                    }
                    for prob, idx in zip(probs, indices)
                ]
            }
            results.append(result)

        return results
```

---

## 3. Model Archiving and Deployment

### 3.1 Creating Model Archive

```bash
# Save model as TorchScript
python -c "
import torch
model = YourModel()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
scripted = torch.jit.script(model)
scripted.save('model.pt')
"

# Create MAR file
torch-model-archiver \
    --model-name my_model \
    --version 1.0 \
    --serialized-file model.pt \
    --handler custom_handler.py \
    --extra-files "index_to_name.json,config.json" \
    --export-path model_store

# Generated file: model_store/my_model.mar
```

### 3.2 Starting TorchServe

```bash
# Basic start
torchserve --start \
    --model-store model_store \
    --models my_model=my_model.mar

# Start with config file
torchserve --start \
    --model-store model_store \
    --models my_model=my_model.mar \
    --ts-config config.properties

# Run with Docker
docker run -d \
    --name torchserve \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8082:8082 \
    -v $(pwd)/model_store:/home/model-server/model-store \
    pytorch/torchserve:latest \
    torchserve --start --model-store /home/model-server/model-store
```

### 3.3 Configuration File

```properties
# config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

# Worker settings
default_workers_per_model=4
job_queue_size=1000

# Batch settings
max_batch_delay=100
batch_size=32

# GPU settings
number_of_gpu=1

# Model settings
model_store=/home/model-server/model-store
load_models=my_model.mar
```

### 3.4 API Calls

```python
"""
TorchServe API calls
"""

import requests
import json

# Prediction request
def predict(data):
    response = requests.post(
        "http://localhost:8080/predictions/my_model",
        json=data
    )
    return response.json()

# Single prediction
result = predict({"data": [1.0, 2.0, 3.0, 4.0]})
print(result)

# Batch prediction
batch_data = [
    {"data": [1.0, 2.0, 3.0, 4.0]},
    {"data": [5.0, 6.0, 7.0, 8.0]}
]
results = [predict(d) for d in batch_data]

# Management API
# List models
models = requests.get("http://localhost:8081/models").json()

# Model details
model_info = requests.get("http://localhost:8081/models/my_model").json()

# Worker scaling
requests.put(
    "http://localhost:8081/models/my_model",
    params={"min_worker": 2, "max_worker": 4}
)

# Register model
requests.post(
    "http://localhost:8081/models",
    params={
        "url": "my_model_v2.mar",
        "initial_workers": 2
    }
)
```

---

## 4. Triton Inference Server

### 4.1 Triton Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Triton Inference Server                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Supported Frameworks:                                             │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│   │ PyTorch  │ │TensorFlow│ │   ONNX   │ │ TensorRT │              │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                           │
│   │  Python  │ │   DALI   │ │   vLLM   │                           │
│   └──────────┘ └──────────┘ └──────────┘                           │
│                                                                     │
│   Key Features:                                                     │
│   - Dynamic Batching                                                │
│   - Model Ensembles                                                 │
│   - Multi-Model Serving                                             │
│   - GPU Scheduling                                                  │
│   - Model Versioning                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Model Repository Structure

```
model_repository/
├── model_a/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.onnx
│   └── 2/
│       └── model.onnx
├── model_b/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
└── ensemble_model/
    └── config.pbtxt
```

### 4.3 Model Configuration

```protobuf
# config.pbtxt - ONNX model
name: "churn_predictor"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [4]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [2]
  }
]

# Multiple instances on the same GPU — overlaps compute and data transfer,
# keeping the GPU busy while one instance waits for I/O
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

# Dynamic batching groups incoming requests into batches — amortizes GPU kernel launch
# overhead across multiple inputs, increasing throughput 3-5x with minimal latency cost
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  # 100us max wait — balances batch size vs latency; increase for throughput-critical workloads
  max_queue_delay_microseconds: 100
}

# Version policy
version_policy: {
  latest: {
    num_versions: 2
  }
}
```

```protobuf
# config.pbtxt - PyTorch model
name: "image_classifier"
platform: "pytorch_libtorch"
max_batch_size: 64

input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [1000]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### 4.4 Running Triton

```bash
# Run with Docker
docker run --gpus all -d \
    --name triton \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models

# Health check
curl -v localhost:8000/v2/health/ready

# Model metadata
curl localhost:8000/v2/models/churn_predictor
```

### 4.5 Python Client

```python
"""
Triton Python client
"""

import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

# HTTP client
def triton_http_inference(model_name: str, input_data: np.ndarray):
    """Inference via HTTP"""
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Setup inputs
    inputs = [
        httpclient.InferInput("input", input_data.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(input_data)

    # Setup outputs
    outputs = [
        httpclient.InferRequestedOutput("output")
    ]

    # Inference
    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )

    return response.as_numpy("output")

# gRPC client (faster)
def triton_grpc_inference(model_name: str, input_data: np.ndarray):
    """Inference via gRPC"""
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    inputs = [
        grpcclient.InferInput("input", input_data.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        grpcclient.InferRequestedOutput("output")
    ]

    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )

    return response.as_numpy("output")

# Example usage
data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
result = triton_http_inference("churn_predictor", data)
print(f"Predictions: {result}")
```

---

## 5. Model Optimization

### 5.1 ONNX Conversion

```python
"""
Convert PyTorch model to ONNX
"""

import torch
import torch.onnx

# Load model
model = YourModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Dummy input
dummy_input = torch.randn(1, 4)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

# Validate ONNX model
import onnx
import onnxruntime as ort

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Test inference
session = ort.InferenceSession("model.onnx")
result = session.run(
    None,
    {"input": dummy_input.numpy()}
)
print(f"ONNX output: {result}")
```

### 5.2 TensorRT Optimization

```python
"""
TensorRT optimization
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def build_engine(onnx_path: str, engine_path: str):
    """Build TensorRT engine from ONNX"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # FP16 optimization (if supported)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Dynamic batch configuration
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 4), (16, 4), (64, 4))  # min, opt, max
    config.add_optimization_profile(profile)

    # Build engine
    engine = builder.build_serialized_network(network, config)

    # Save
    with open(engine_path, "wb") as f:
        f.write(engine)

    return engine

# Build engine
build_engine("model.onnx", "model.engine")
```

### 5.3 Quantization

```python
"""
Model quantization
"""

import torch
from torch.quantization import quantize_dynamic, quantize_static

# Dynamic quantization (simplest)
model_int8 = quantize_dynamic(
    model,
    {torch.nn.Linear},  # Layer types to quantize
    dtype=torch.qint8
)

# Static quantization (better performance)
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
model_prepared = torch.quantization.prepare(model)

# Calibration (with representative data)
with torch.no_grad():
    for data in calibration_data:
        model_prepared(data)

model_quantized = torch.quantization.convert(model_prepared)

# Size comparison
import os
torch.save(model.state_dict(), "model_fp32.pth")
torch.save(model_quantized.state_dict(), "model_int8.pth")

print(f"FP32 size: {os.path.getsize('model_fp32.pth') / 1e6:.2f} MB")
print(f"INT8 size: {os.path.getsize('model_int8.pth') / 1e6:.2f} MB")
```

---

## 6. Multi-Model Serving

### 6.1 Triton Ensembles

```protobuf
# ensemble_model/config.pbtxt
name: "ensemble_pipeline"
platform: "ensemble"
max_batch_size: 32

input [
  {
    name: "raw_input"
    data_type: TYPE_FP32
    dims: [4]
  }
]

output [
  {
    name: "final_output"
    data_type: TYPE_FP32
    dims: [2]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessor"
      model_version: -1
      input_map {
        key: "raw_input"
        value: "raw_input"
      }
      output_map {
        key: "processed_output"
        value: "processed"
      }
    },
    {
      model_name: "classifier"
      model_version: -1
      input_map {
        key: "processed"
        value: "input"
      }
      output_map {
        key: "output"
        value: "final_output"
      }
    }
  ]
}
```

### 6.2 A/B Testing

```python
"""
TorchServe A/B testing setup
"""

import random
import requests

class ABTestRouter:
    """A/B test router"""

    def __init__(self, model_a: str, model_b: str, traffic_split: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.base_url = "http://localhost:8080/predictions"

    def predict(self, data: dict) -> dict:
        """A/B distributed prediction"""
        # Traffic distribution
        if random.random() < self.traffic_split:
            model = self.model_a
            variant = "A"
        else:
            model = self.model_b
            variant = "B"

        # Prediction
        response = requests.post(
            f"{self.base_url}/{model}",
            json=data
        )

        result = response.json()
        result["variant"] = variant
        result["model"] = model

        return result

# Usage
router = ABTestRouter("model_v1", "model_v2", traffic_split=0.8)
result = router.predict({"data": [1.0, 2.0, 3.0, 4.0]})
```

---

## Practice Exercises

### Exercise 1: TorchServe Deployment
Deploy a PyTorch image classification model with TorchServe.

### Exercise 2: ONNX Conversion
Convert a PyTorch model to ONNX and serve it with Triton.

### Exercise 3: Performance Optimization
Optimize a model with TensorRT and compare inference speed.

---

## Summary

| Tool | Advantages | Best For |
|------|-----------|----------|
| TorchServe | PyTorch native, simple | PyTorch models |
| Triton | Multi-framework, high performance | Complex requirements |
| ONNX Runtime | Universal, cross-platform | Lightweight deployment |
| TensorRT | GPU optimization | Maximum performance |

---

## References

- [TorchServe Documentation](https://pytorch.org/serve/)
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT](https://developer.nvidia.com/tensorrt)
