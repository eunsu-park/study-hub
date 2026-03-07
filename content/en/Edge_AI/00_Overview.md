# Edge AI Study Guide

## Introduction

This folder provides a comprehensive guide to **Edge AI** — the practice of deploying machine learning models directly on edge devices such as smartphones, microcontrollers, embedded systems, and IoT sensors. Edge AI eliminates the need to send data to the cloud for inference, enabling real-time decisions with lower latency, improved privacy, and reduced bandwidth costs.

The curriculum covers the full pipeline from model compression and optimization to hardware-specific deployment, spanning both theoretical foundations and hands-on implementation with PyTorch, ONNX, TensorFlow Lite, and vendor-specific toolchains.

## Target Audience

- Learners who have completed the **Deep_Learning** folder (or equivalent knowledge of CNNs, Transformers, and training workflows)
- Engineers interested in deploying models on resource-constrained devices
- Researchers exploring efficient model design and hardware-aware optimization
- Anyone building real-time AI applications (robotics, autonomous vehicles, smart cameras, wearables)

## Prerequisites

- **Deep_Learning**: Solid understanding of CNNs, training loops, loss functions, and PyTorch
- **Computer_Architecture**: Familiarity with CPU/GPU pipelines, memory hierarchy, and instruction-level parallelism
- **IoT_Embedded**: Basic knowledge of embedded systems, microcontrollers, and sensor interfaces

## Learning Roadmap

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Foundations     │────▶│   Compression    │────▶│   Optimization   │
│     L01-L02       │     │     L03-L05      │     │     L06-L07      │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    Practical      │◀────│    Deployment    │◀────│     Export &     │
│     L14-L16       │     │     L10-L13      │     │    Runtimes      │
└──────────────────┘     └──────────────────┘     │     L08-L09      │
                                                  └──────────────────┘
```

**Recommended Path**:
1. Start with Foundations (L01-L02) to understand edge AI constraints and compression taxonomy
2. Master Compression techniques (L03-L05) — quantization, pruning, and knowledge distillation
3. Study Efficient Architecture design (L06-L07) — MobileNet, EfficientNet, and NAS
4. Learn Export & Runtimes (L08-L09) — ONNX, TensorFlow Lite, and inference engines
5. Explore Hardware Deployment (L10-L13) — TensorRT, mobile/MCU targets, and benchmarking
6. Apply knowledge with Practical projects (L14-L16) — end-to-end edge AI applications

## File List

| Lesson | Filename | Difficulty | Description |
|--------|----------|------------|-------------|
| **Block 1: Foundations** |
| L01 | `01_Edge_AI_Fundamentals.md` | ⭐ | Edge vs cloud inference, latency/privacy tradeoffs, edge computing spectrum |
| L02 | `02_Model_Compression_Overview.md` | ⭐⭐ | Compression taxonomy: pruning, quantization, distillation, NAS |
| **Block 2: Compression Techniques** |
| L03 | `03_Quantization.md` | ⭐⭐⭐ | PTQ vs QAT, INT8/INT4, symmetric vs asymmetric, mixed-precision |
| L04 | `04_Pruning.md` | ⭐⭐⭐ | Structured vs unstructured, magnitude-based, lottery ticket hypothesis |
| L05 | `05_Knowledge_Distillation.md` | ⭐⭐⭐ | Teacher-student framework, soft targets, attention transfer |
| **Block 3: Efficient Architecture Design** |
| L06 | `06_Efficient_Architectures.md` | ⭐⭐⭐ | MobileNet, EfficientNet, ShuffleNet, SqueezeNet, design principles |
| L07 | `07_Neural_Architecture_Search.md` | ⭐⭐⭐⭐ | NAS fundamentals, search strategies, hardware-aware NAS |
| **Block 4: Export and Runtimes** |
| L08 | `08_ONNX_and_Model_Export.md` | ⭐⭐⭐ | ONNX format, graph optimization, cross-framework conversion |
| L09 | `09_TFLite_and_Mobile_Runtimes.md` | ⭐⭐⭐ | TensorFlow Lite, CoreML, NNAPI, delegate system |
| **Block 5: Hardware Deployment** |
| L10 | `10_TensorRT_Optimization.md` | ⭐⭐⭐⭐ | NVIDIA TensorRT, layer fusion, INT8 calibration, engine building |
| L11 | `11_Mobile_Deployment.md` | ⭐⭐⭐ | Android (NNAPI), iOS (CoreML), on-device inference pipelines |
| L12 | `12_MCU_Deployment.md` | ⭐⭐⭐⭐ | TinyML, TFLite Micro, CMSIS-NN, memory-constrained inference |
| L13 | `13_Edge_Hardware_Landscape.md` | ⭐⭐⭐ | NPUs, TPU Edge, Jetson, Coral, Hailo, hardware comparison |
| **Block 6: Practical Applications** |
| L14 | `14_Benchmarking_and_Profiling.md` | ⭐⭐⭐ | Latency/throughput measurement, power profiling, roofline analysis |
| L15 | `15_Practical_Edge_Vision.md` | ⭐⭐⭐⭐ | End-to-end: object detection on Raspberry Pi / Jetson Nano |
| L16 | `16_Practical_Edge_NLP.md` | ⭐⭐⭐⭐ | End-to-end: on-device text classification and keyword spotting |

**Total: 16 lessons** (13 concept lessons + 3 practical/implementation lessons)

## Environment Setup

### Core Installation

```bash
# PyTorch (for model training and export)
pip install torch torchvision

# ONNX ecosystem
pip install onnx onnxruntime onnxoptimizer

# TensorFlow Lite (for mobile/MCU deployment)
pip install tensorflow tflite-runtime

# Profiling and benchmarking
pip install thop fvcore
```

### Optional Tools

```bash
# NVIDIA TensorRT (requires NVIDIA GPU + CUDA)
pip install tensorrt

# Edge TPU compiler (for Google Coral)
# See: https://coral.ai/docs/edgetpu/compiler/

# ARM NN SDK (for ARM-based deployment)
# See: https://developer.arm.com/Tools%20and%20Software/Arm%20NN
```

### Verify Installation

```python
import torch
import onnx
import onnxruntime as ort

print(f"PyTorch: {torch.__version__}")
print(f"ONNX: {onnx.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Related Topics

- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: Prerequisite — CNN architectures, training, and PyTorch fundamentals
- **[Computer_Vision](../Computer_Vision/00_Overview.md)**: Applied vision tasks (detection, segmentation) commonly deployed on edge
- **[IoT_Embedded](../IoT_Embedded/00_Overview.md)**: Embedded systems, sensors, and microcontroller programming
- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: Model compression at scale — LoRA, quantization for LLMs
- **[Computer_Architecture](../Computer_Architecture/00_Overview.md)**: Hardware fundamentals — memory hierarchy, parallelism, accelerators

## Study Tips

1. **Start with Profiling**: Before compressing a model, measure its baseline latency and memory on your target device
2. **Compression is Iterative**: Apply one technique at a time (quantize, then prune, then distill) and measure impact
3. **Hardware Matters**: A technique that works well on GPU (e.g., unstructured pruning) may not help on mobile NPUs
4. **Test on Real Devices**: Emulators and simulators cannot capture true latency — always validate on hardware
5. **Read Vendor Docs**: Each hardware platform (TensorRT, CoreML, Edge TPU) has specific operator support and constraints
6. **Track Accuracy vs Efficiency**: Build Pareto curves (accuracy vs latency/size) to find the sweet spot

## Learning Outcomes

After completing this folder, you will be able to:

- Explain the tradeoffs between cloud inference and edge deployment
- Apply quantization (PTQ, QAT) to reduce model size by 2-4x with minimal accuracy loss
- Prune neural networks using structured and unstructured methods
- Train compact student models via knowledge distillation
- Design efficient architectures using MobileNet, EfficientNet, and NAS principles
- Export models to ONNX and TensorFlow Lite formats
- Deploy optimized models on GPUs (TensorRT), mobile devices, and microcontrollers
- Benchmark and profile edge inference pipelines for latency, throughput, and power

## Next Steps

- **For LLM Compression**: Proceed to `Foundation_Models` for quantization and LoRA on large language models
- **For Vision Applications**: Explore `Computer_Vision` for detection, tracking, and SLAM on edge
- **For Production Pipelines**: Check `MLOps` for model serving, monitoring, and CI/CD
- **For Hardware Design**: Study `Computer_Architecture` for custom accelerator design

---

**License**: CC BY-NC 4.0

Start with [01_Edge_AI_Fundamentals.md](./01_Edge_AI_Fundamentals.md) to begin your edge AI journey.
