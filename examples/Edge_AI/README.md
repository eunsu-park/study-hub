# Edge_AI Examples

Runnable example code corresponding to the lessons in the Edge_AI folder.

## File List

| # | Filename | Lesson | Description |
|---|----------|--------|-------------|
| 01 | `01_quantization_basics.py` | L03 | PyTorch dynamic and static quantization (INT8), per-tensor vs per-channel |
| 02 | `02_pruning_tutorial.py` | L04 | Magnitude-based, structured, and iterative pruning with fine-tuning |
| 03 | `03_knowledge_distillation.py` | L05 | Teacher-student training loop with temperature scaling and KD loss |
| 04 | `04_efficient_architectures.py` | L06 | MobileNet depthwise separable conv, FLOPs comparison with standard conv |
| 05 | `05_onnx_export.py` | L08 | Export PyTorch model to ONNX, optimize graph, run with ONNX Runtime |
| 06 | `06_tflite_conversion.py` | L09 | TensorFlow model to TFLite conversion with quantization options |
| 07 | `07_model_profiling.py` | L14 | Parameter count, FLOPs estimation, memory footprint, inference timing |
| 08 | `08_edge_inference_pipeline.py` | L15 | Complete edge inference pipeline: load, preprocess, infer, postprocess |

## How to Run

### Environment Setup

```bash
# Create virtual environment
python -m venv edge-ai-env
source edge-ai-env/bin/activate

# Core dependencies
pip install torch torchvision
pip install onnx onnxruntime

# Optional: TensorFlow/TFLite (for 06_tflite_conversion.py)
pip install tensorflow

# Profiling utilities
pip install thop fvcore
```

### Execution

```bash
cd Edge_AI/examples
python 01_quantization_basics.py
python 02_pruning_tutorial.py
# ... etc
```

## Learning Path

### Stage 1: Model Compression (Examples 01-03)
```
01_quantization_basics.py     # Reduce precision: FP32 -> INT8
02_pruning_tutorial.py        # Remove redundant weights
03_knowledge_distillation.py  # Train smaller student from larger teacher
```

### Stage 2: Efficient Design (Example 04)
```
04_efficient_architectures.py # Design models that are inherently efficient
```

### Stage 3: Export and Runtime (Examples 05-06)
```
05_onnx_export.py             # Cross-framework model export
06_tflite_conversion.py       # Mobile/embedded deployment format
```

### Stage 4: Profiling and Deployment (Examples 07-08)
```
07_model_profiling.py         # Measure what matters: latency, FLOPs, memory
08_edge_inference_pipeline.py # End-to-end inference with timing breakdown
```

## Prerequisites

- Python 3.8+
- PyTorch 2.0+ (for quantization and pruning APIs)
- ONNX Runtime 1.14+ (for example 05)
- TensorFlow 2.13+ (for example 06, optional)

## References

- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileNet Paper](https://arxiv.org/abs/1704.04861)
