# Edge_AI Exercises

Practice problem solutions for the Edge_AI lessons. Each file contains 3-5 exercises
with working solutions that can be run independently.

## File List

| # | Filename | Lesson | Description |
|---|----------|--------|-------------|
| 01 | `01_edge_ai_fundamentals.py` | L01 | Edge vs cloud tradeoffs, latency estimation, deployment decision framework |
| 02 | `02_model_compression_overview.py` | L02 | Compression taxonomy, technique selection, compression ratio calculation |
| 03 | `03_quantization.py` | L03 | PTQ vs QAT, scale/zero-point computation, mixed-precision strategy |
| 04 | `04_pruning.py` | L04 | Magnitude pruning, structured pruning, lottery ticket analysis |
| 05 | `05_knowledge_distillation.py` | L05 | Temperature scaling, KD loss implementation, dark knowledge analysis |
| 06 | `06_efficient_architectures.py` | L06 | Depthwise separable conv FLOPs, MobileNet V2, width multiplier |
| 07 | `07_neural_architecture_search.py` | L07 | Search space design, NAS evaluation, hardware-aware objectives |
| 08 | `08_onnx_and_model_export.py` | L08 | ONNX export, graph inspection, operator compatibility |
| 09 | `09_tflite_and_mobile_runtimes.py` | L09 | TFLite conversion, delegate selection, runtime comparison |
| 10 | `10_tensorrt_optimization.py` | L10 | TensorRT concepts, layer fusion, INT8 calibration strategy |
| 11 | `11_mobile_deployment.py` | L11 | Mobile pipeline design, NNAPI/CoreML selection, memory budgeting |
| 12 | `12_mcu_deployment.py` | L12 | TinyML constraints, memory layout, CMSIS-NN operator mapping |
| 13 | `13_edge_hardware_landscape.py` | L13 | Hardware comparison, NPU/TPU tradeoffs, deployment matching |
| 14 | `14_benchmarking_and_profiling.py` | L14 | Latency measurement, roofline analysis, power estimation |
| 15 | `15_practical_edge_vision.py` | L15 | Object detection pipeline, NMS implementation, Jetson deployment |
| 16 | `16_practical_edge_nlp.py` | L16 | On-device text classification, keyword spotting, model selection |

## How to Run

```bash
cd exercises/Edge_AI
python 01_edge_ai_fundamentals.py
python 02_model_compression_overview.py
# ... etc
```

## Prerequisites

```bash
pip install torch torchvision numpy
pip install onnx onnxruntime  # for exercises 08
```
