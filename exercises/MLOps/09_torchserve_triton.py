"""
Exercise Solutions: TorchServe and Triton
===========================================
Lesson 09 from MLOps topic.

Exercises
---------
1. TorchServe Deployment — Implement a custom TorchServe handler with
   preprocessing, inference, and postprocessing logic.
2. ONNX Conversion — Simulate model export to ONNX format with
   optimization passes and validation.
3. Performance Optimization — Implement dynamic batching, quantization
   simulation, and benchmark comparisons.
"""

import math
import random
import time
import json
from datetime import datetime


# ============================================================
# Exercise 1: TorchServe Deployment
# ============================================================

def exercise_1_torchserve_handler():
    """Implement a custom TorchServe handler.

    A TorchServe handler implements 4 methods:
    - initialize(): Load model and setup
    - preprocess(): Transform raw input to model-ready format
    - inference(): Run the model
    - postprocess(): Transform model output to response format

    We simulate this pattern for an image classification model.
    """

    class ImageClassificationHandler:
        """Custom TorchServe handler for image classification.

        In real TorchServe, this would extend BaseHandler and work with
        PyTorch models. Here we simulate the full lifecycle.
        """

        def __init__(self):
            self.model = None
            self.class_names = []
            self.initialized = False
            self.device = "cpu"  # Would be "cuda" in production

        def initialize(self, context=None):
            """Load model and class labels.

            In real TorchServe:
            - context.system_properties provides model_dir, gpu_id
            - Model is loaded from the .mar archive
            - Class names from index_to_name.json
            """
            print("  [initialize] Loading model...")

            # Simulate model loading
            random.seed(42)
            n_classes = 10
            n_features = 64  # Simulated flattened features
            self.model = {
                "weights": [[random.gauss(0, 0.1) for _ in range(n_features)]
                            for _ in range(n_classes)],
                "biases": [random.gauss(0, 0.01) for _ in range(n_classes)],
            }

            self.class_names = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ]

            self.initialized = True
            print(f"  [initialize] Model loaded: {n_classes} classes, "
                  f"{n_features} features, device={self.device}")

        def preprocess(self, data):
            """Transform raw input into model-ready format.

            In real TorchServe:
            - Decode image bytes (JPEG/PNG)
            - Resize to model input size (e.g., 224x224)
            - Normalize with ImageNet mean/std
            - Convert to torch.Tensor

            Here we simulate the preprocessing pipeline.
            """
            preprocessed = []
            for item in data:
                # Simulate image data as random features
                raw = item.get("body", item.get("data", {}))

                if isinstance(raw, dict):
                    image_size = raw.get("image_size", [32, 32, 3])
                    pixels = raw.get("pixels", None)
                else:
                    image_size = [32, 32, 3]
                    pixels = None

                # Simulate preprocessing steps
                steps = []

                # 1. Resize (bilinear interpolation simulation)
                target_size = [224, 224]
                steps.append(f"resize {image_size[:2]} -> {target_size}")

                # 2. Center crop
                steps.append(f"center_crop 224x224")

                # 3. Normalize (ImageNet stats)
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                steps.append(f"normalize mean={mean} std={std}")

                # 4. Flatten to feature vector (simulate feature extraction)
                features = [random.gauss(0, 1) for _ in range(64)]
                steps.append(f"extract_features -> {len(features)} dims")

                preprocessed.append({
                    "features": features,
                    "preprocessing_steps": steps,
                })

            return preprocessed

        def inference(self, data):
            """Run model inference.

            In real TorchServe:
            - Input is a batch of tensors
            - Uses torch.no_grad() for inference mode
            - Returns raw logits
            """
            results = []
            for item in data:
                features = item["features"]
                # Simple linear classifier (simulates neural network output)
                logits = []
                for class_idx in range(len(self.class_names)):
                    w = self.model["weights"][class_idx]
                    b = self.model["biases"][class_idx]
                    logit = sum(wi * fi for wi, fi in zip(w, features)) + b
                    logits.append(logit)
                results.append(logits)
            return results

        def postprocess(self, inference_output, top_k=5):
            """Transform raw model output to human-readable response.

            Converts logits to probabilities via softmax, returns top-K
            predictions with class names and confidence scores.
            """
            responses = []
            for logits in inference_output:
                # Softmax
                max_logit = max(logits)
                exp_logits = [math.exp(l - max_logit) for l in logits]
                sum_exp = sum(exp_logits)
                probabilities = [e / sum_exp for e in exp_logits]

                # Top-K predictions
                indexed_probs = list(enumerate(probabilities))
                indexed_probs.sort(key=lambda x: -x[1])
                top_predictions = []
                for idx, prob in indexed_probs[:top_k]:
                    top_predictions.append({
                        "class_name": self.class_names[idx],
                        "class_index": idx,
                        "confidence": round(prob, 6),
                    })

                responses.append({
                    "top_k": top_predictions,
                    "predicted_class": top_predictions[0]["class_name"],
                    "confidence": top_predictions[0]["confidence"],
                })
            return responses

        def handle(self, data):
            """Full request handling pipeline."""
            if not self.initialized:
                self.initialize()

            start = time.time()
            preprocessed = self.preprocess(data)
            preprocess_time = time.time() - start

            start = time.time()
            inference_output = self.inference(preprocessed)
            inference_time = time.time() - start

            start = time.time()
            responses = self.postprocess(inference_output)
            postprocess_time = time.time() - start

            # Add timing metadata
            for resp in responses:
                resp["timing"] = {
                    "preprocess_ms": round(preprocess_time * 1000, 3),
                    "inference_ms": round(inference_time * 1000, 3),
                    "postprocess_ms": round(postprocess_time * 1000, 3),
                    "total_ms": round((preprocess_time + inference_time + postprocess_time) * 1000, 3),
                }
            return responses

    # --- Test the handler ---
    print("TorchServe Custom Handler")
    print("=" * 60)

    handler = ImageClassificationHandler()
    handler.initialize()

    # Single image request
    print("\n  Single Image Prediction:")
    print("-" * 40)
    request_data = [{"body": {"image_size": [32, 32, 3]}}]
    results = handler.handle(request_data)
    for result in results:
        print(f"    Predicted: {result['predicted_class']} "
              f"(confidence: {result['confidence']:.4f})")
        print(f"    Top-5:")
        for pred in result["top_k"]:
            bar = "█" * int(pred["confidence"] * 40)
            print(f"      {pred['class_name']:<12s} {pred['confidence']:.4f} {bar}")
        print(f"    Timing: {result['timing']}")

    # Batch request
    print("\n  Batch Prediction (3 images):")
    print("-" * 40)
    batch_data = [{"body": {"image_size": [32, 32, 3]}} for _ in range(3)]
    results = handler.handle(batch_data)
    for i, result in enumerate(results):
        print(f"    Image {i}: {result['predicted_class']} "
              f"(confidence: {result['confidence']:.4f})")

    # Handler configuration (what would go in model-config.yaml)
    print("\n  Handler Configuration:")
    print("-" * 40)
    config = {
        "handler": "image_classification_handler.py",
        "model_name": "resnet50",
        "serialized_file": "model.pt",
        "extra_files": "index_to_name.json",
        "batch_size": 8,
        "max_batch_delay": 100,  # ms
        "response_timeout": 120,  # seconds
        "device_type": "gpu",
    }
    print(f"  {json.dumps(config, indent=4)}")

    return handler


# ============================================================
# Exercise 2: ONNX Conversion
# ============================================================

def exercise_2_onnx_conversion():
    """Simulate model export to ONNX format with optimizations.

    ONNX (Open Neural Network Exchange) enables:
    - Framework-independent model representation
    - Optimization passes (constant folding, operator fusion)
    - Runtime acceleration (ONNX Runtime, TensorRT)
    - Cross-platform deployment
    """

    class ONNXModel:
        """Simulated ONNX model representation."""

        def __init__(self, name, opset_version=17):
            self.name = name
            self.opset_version = opset_version
            self.graph = {"nodes": [], "inputs": [], "outputs": []}
            self.metadata = {}
            self.optimized = False
            self.size_bytes = 0

        def add_input(self, name, shape, dtype="float32"):
            self.graph["inputs"].append({
                "name": name, "shape": shape, "dtype": dtype
            })

        def add_output(self, name, shape, dtype="float32"):
            self.graph["outputs"].append({
                "name": name, "shape": shape, "dtype": dtype
            })

        def add_node(self, op_type, inputs, outputs, attributes=None):
            self.graph["nodes"].append({
                "op_type": op_type,
                "inputs": inputs,
                "outputs": outputs,
                "attributes": attributes or {},
            })

        def apply_optimization(self, optimization_name, size_reduction_pct=0):
            """Apply an optimization pass."""
            original_nodes = len(self.graph["nodes"])
            if optimization_name == "constant_folding":
                # Remove constant computation nodes
                self.graph["nodes"] = [
                    n for n in self.graph["nodes"]
                    if n["op_type"] not in ["Constant", "Shape"]
                ]
            elif optimization_name == "operator_fusion":
                # Fuse consecutive operations (e.g., Conv+BN+ReLU)
                fused_nodes = []
                i = 0
                while i < len(self.graph["nodes"]):
                    node = self.graph["nodes"][i]
                    # Fuse Conv + BatchNorm + Relu
                    if (node["op_type"] == "Conv" and
                            i + 2 < len(self.graph["nodes"]) and
                            self.graph["nodes"][i + 1]["op_type"] == "BatchNormalization" and
                            self.graph["nodes"][i + 2]["op_type"] == "Relu"):
                        fused_nodes.append({
                            "op_type": "FusedConvBNRelu",
                            "inputs": node["inputs"],
                            "outputs": self.graph["nodes"][i + 2]["outputs"],
                            "attributes": {**node["attributes"], "fused": True},
                        })
                        i += 3
                    else:
                        fused_nodes.append(node)
                        i += 1
                self.graph["nodes"] = fused_nodes
            elif optimization_name == "eliminate_identity":
                self.graph["nodes"] = [
                    n for n in self.graph["nodes"]
                    if n["op_type"] != "Identity"
                ]

            new_nodes = len(self.graph["nodes"])
            self.size_bytes = int(self.size_bytes * (1 - size_reduction_pct / 100))
            return {
                "optimization": optimization_name,
                "nodes_before": original_nodes,
                "nodes_after": new_nodes,
                "nodes_removed": original_nodes - new_nodes,
            }

    def export_to_onnx(model_name, input_shape, architecture):
        """Simulate exporting a PyTorch model to ONNX."""
        onnx_model = ONNXModel(model_name, opset_version=17)

        # Define input/output
        onnx_model.add_input("images", ["batch", *input_shape])
        onnx_model.add_output("logits", ["batch", architecture["num_classes"]])

        # Build graph from architecture
        for layer in architecture["layers"]:
            onnx_model.add_node(
                op_type=layer["type"],
                inputs=[layer.get("input", "prev")],
                outputs=[layer["name"]],
                attributes=layer.get("attrs", {}),
            )

        # Calculate size
        onnx_model.size_bytes = architecture.get("size_mb", 50) * 1024 * 1024

        return onnx_model

    # --- Define a ResNet-like architecture ---
    architecture = {
        "num_classes": 1000,
        "size_mb": 97.8,
        "layers": [
            {"type": "Conv", "name": "conv1", "attrs": {"kernel": [7, 7], "stride": 2}},
            {"type": "BatchNormalization", "name": "bn1"},
            {"type": "Relu", "name": "relu1"},
            {"type": "MaxPool", "name": "pool1", "attrs": {"kernel": [3, 3], "stride": 2}},
            # Block 1
            {"type": "Conv", "name": "block1_conv1", "attrs": {"kernel": [3, 3]}},
            {"type": "BatchNormalization", "name": "block1_bn1"},
            {"type": "Relu", "name": "block1_relu1"},
            {"type": "Conv", "name": "block1_conv2", "attrs": {"kernel": [3, 3]}},
            {"type": "BatchNormalization", "name": "block1_bn2"},
            {"type": "Add", "name": "block1_residual"},
            {"type": "Relu", "name": "block1_relu2"},
            # Block 2
            {"type": "Conv", "name": "block2_conv1", "attrs": {"kernel": [3, 3]}},
            {"type": "BatchNormalization", "name": "block2_bn1"},
            {"type": "Relu", "name": "block2_relu1"},
            {"type": "Identity", "name": "identity1"},
            # Classifier
            {"type": "GlobalAveragePool", "name": "gap"},
            {"type": "Flatten", "name": "flatten"},
            {"type": "Gemm", "name": "fc", "attrs": {"units": 1000}},
            {"type": "Constant", "name": "const1"},
            {"type": "Shape", "name": "shape1"},
        ],
    }

    print("ONNX Model Conversion")
    print("=" * 60)

    # Step 1: Export
    print("\n1. Export PyTorch Model to ONNX")
    print("-" * 40)
    onnx_model = export_to_onnx("resnet50", [3, 224, 224], architecture)
    print(f"  Model: {onnx_model.name}")
    print(f"  Opset: {onnx_model.opset_version}")
    print(f"  Nodes: {len(onnx_model.graph['nodes'])}")
    print(f"  Size: {onnx_model.size_bytes / (1024*1024):.1f} MB")
    print(f"  Input: {onnx_model.graph['inputs']}")
    print(f"  Output: {onnx_model.graph['outputs']}")

    # Step 2: Validate
    print("\n2. Model Validation")
    print("-" * 40)
    validations = [
        ("Opset version >= 13", onnx_model.opset_version >= 13),
        ("Input shape defined", len(onnx_model.graph["inputs"]) > 0),
        ("Output shape defined", len(onnx_model.graph["outputs"]) > 0),
        ("No unsupported ops", True),  # Simplified
        ("Dynamic batch support", "batch" in str(onnx_model.graph["inputs"][0]["shape"])),
    ]
    for check, passed in validations:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    # Step 3: Optimize
    print("\n3. Optimization Passes")
    print("-" * 40)
    optimizations = [
        ("constant_folding", 2),
        ("eliminate_identity", 0),
        ("operator_fusion", 15),
    ]
    for opt_name, size_reduction in optimizations:
        result = onnx_model.apply_optimization(opt_name, size_reduction)
        print(f"  {opt_name}:")
        print(f"    Nodes: {result['nodes_before']} -> {result['nodes_after']} "
              f"(-{result['nodes_removed']})")

    print(f"\n  Final model: {len(onnx_model.graph['nodes'])} nodes, "
          f"{onnx_model.size_bytes / (1024*1024):.1f} MB")

    # Step 4: Runtime comparison
    print("\n4. Runtime Comparison (simulated)")
    print("-" * 40)
    runtimes = [
        {"name": "PyTorch (native)", "latency_ms": 45.2, "throughput": 22},
        {"name": "ONNX Runtime (CPU)", "latency_ms": 28.1, "throughput": 35},
        {"name": "ONNX Runtime (GPU)", "latency_ms": 8.3, "throughput": 120},
        {"name": "TensorRT (GPU)", "latency_ms": 4.1, "throughput": 244},
    ]
    print(f"  {'Runtime':<25s} {'Latency':>10s} {'Throughput':>12s} {'Speedup':>8s}")
    print(f"  {'-'*55}")
    base_latency = runtimes[0]["latency_ms"]
    for rt in runtimes:
        speedup = base_latency / rt["latency_ms"]
        print(f"  {rt['name']:<25s} {rt['latency_ms']:>8.1f}ms "
              f"{rt['throughput']:>8d} img/s {speedup:>7.1f}x")

    return onnx_model


# ============================================================
# Exercise 3: Performance Optimization
# ============================================================

def exercise_3_performance_optimization():
    """Implement dynamic batching, quantization, and benchmarks.

    Key optimization techniques:
    - Dynamic batching: Accumulate requests into batches for GPU efficiency
    - Quantization: Reduce precision (FP32 -> INT8) for faster inference
    - Model pruning: Remove near-zero weights
    """

    # --- Dynamic Batching ---
    class DynamicBatcher:
        """Accumulates incoming requests into batches for efficient inference.

        Why batch? GPU throughput scales sublinearly with batch size:
        - Batch=1: 5ms/request (200 req/s)
        - Batch=8: 12ms/batch (667 req/s) — 3.3x improvement

        The batcher waits up to max_delay_ms for enough requests to fill
        a batch, then runs inference on whatever has accumulated.
        """

        def __init__(self, max_batch_size=8, max_delay_ms=100):
            self.max_batch_size = max_batch_size
            self.max_delay_ms = max_delay_ms
            self.queue = []
            self.batches_processed = 0
            self.total_requests = 0
            self.batch_sizes = []

        def add_request(self, request):
            """Add a request to the queue."""
            self.queue.append({
                "request": request,
                "arrival_time": time.time(),
            })
            self.total_requests += 1

        def should_dispatch(self):
            """Check if we should dispatch the current batch."""
            if not self.queue:
                return False
            if len(self.queue) >= self.max_batch_size:
                return True
            # Check timeout
            oldest = self.queue[0]["arrival_time"]
            elapsed_ms = (time.time() - oldest) * 1000
            return elapsed_ms >= self.max_delay_ms

        def dispatch_batch(self):
            """Get the next batch to process."""
            batch_size = min(len(self.queue), self.max_batch_size)
            batch = [self.queue.pop(0) for _ in range(batch_size)]
            self.batches_processed += 1
            self.batch_sizes.append(batch_size)
            return batch

        def get_stats(self):
            return {
                "total_requests": self.total_requests,
                "batches_processed": self.batches_processed,
                "avg_batch_size": (sum(self.batch_sizes) / len(self.batch_sizes)
                                   if self.batch_sizes else 0),
                "max_batch_size_used": max(self.batch_sizes) if self.batch_sizes else 0,
                "batch_efficiency": (sum(self.batch_sizes) /
                                     (self.batches_processed * self.max_batch_size)
                                     if self.batches_processed else 0),
            }

    # --- Quantization Simulation ---
    class QuantizationSimulator:
        """Simulate the effects of model quantization.

        FP32 -> FP16: ~2x memory reduction, minimal accuracy loss
        FP32 -> INT8: ~4x memory reduction, slight accuracy loss
        FP32 -> INT4: ~8x memory reduction, noticeable accuracy loss
        """

        @staticmethod
        def quantize(weights, target_dtype="int8"):
            dtype_info = {
                "fp16": {"bits": 16, "range": (-65504, 65504), "acc_loss": 0.001},
                "int8": {"bits": 8, "range": (-128, 127), "acc_loss": 0.005},
                "int4": {"bits": 4, "range": (-8, 7), "acc_loss": 0.02},
            }

            info = dtype_info[target_dtype]
            w_min, w_max = min(weights), max(weights)

            # Scale and quantize
            scale = (w_max - w_min) / (info["range"][1] - info["range"][0])
            zero_point = info["range"][0] - round(w_min / scale) if scale != 0 else 0

            quantized = []
            for w in weights:
                q = round(w / scale) + zero_point if scale != 0 else 0
                q = max(info["range"][0], min(info["range"][1], q))
                quantized.append(q)

            # Dequantize to measure error
            dequantized = [(q - zero_point) * scale for q in quantized]
            errors = [abs(orig - deq) for orig, deq in zip(weights, dequantized)]
            mean_error = sum(errors) / len(errors)
            max_error = max(errors)

            return {
                "dtype": target_dtype,
                "bits": info["bits"],
                "compression_ratio": 32 / info["bits"],
                "n_weights": len(weights),
                "scale": round(scale, 8),
                "zero_point": zero_point,
                "mean_quantization_error": round(mean_error, 8),
                "max_quantization_error": round(max_error, 8),
                "estimated_accuracy_loss": info["acc_loss"],
            }

    # --- Run benchmarks ---
    print("Performance Optimization")
    print("=" * 60)

    # 1. Dynamic Batching
    print("\n1. Dynamic Batching")
    print("-" * 40)

    configs = [
        {"max_batch": 1, "label": "No batching"},
        {"max_batch": 4, "label": "Batch=4"},
        {"max_batch": 8, "label": "Batch=8"},
        {"max_batch": 16, "label": "Batch=16"},
        {"max_batch": 32, "label": "Batch=32"},
    ]

    print(f"  Simulating 100 requests with varying batch sizes:")
    print(f"  {'Config':<20s} {'Batches':>8s} {'Avg Batch':>10s} {'Efficiency':>11s} "
          f"{'Est. Speedup':>13s}")
    print(f"  {'-'*62}")

    for config in configs:
        batcher = DynamicBatcher(max_batch_size=config["max_batch"], max_delay_ms=50)
        random.seed(42)

        for _ in range(100):
            batcher.add_request({"features": [random.gauss(0, 1) for _ in range(8)]})
            if batcher.should_dispatch():
                batcher.dispatch_batch()
        # Flush remaining
        while batcher.queue:
            batcher.dispatch_batch()

        stats = batcher.get_stats()
        # Speedup: batch processing is faster due to GPU parallelism
        # Diminishing returns after batch fills GPU
        speedup = min(config["max_batch"] * 0.7, 8.0) if config["max_batch"] > 1 else 1.0
        print(f"  {config['label']:<20s} {stats['batches_processed']:>8d} "
              f"{stats['avg_batch_size']:>10.1f} {stats['batch_efficiency']:>10.1%} "
              f"{speedup:>12.1f}x")

    # 2. Quantization
    print("\n2. Quantization Comparison")
    print("-" * 40)

    random.seed(42)
    sample_weights = [random.gauss(0, 0.5) for _ in range(10000)]
    original_size_mb = len(sample_weights) * 4 / (1024 * 1024)  # FP32

    quant_sim = QuantizationSimulator()
    print(f"  Original: FP32, {original_size_mb:.2f} MB ({len(sample_weights)} weights)")
    print()
    print(f"  {'Dtype':<8s} {'Bits':>5s} {'Compress':>9s} {'Size MB':>8s} "
          f"{'Mean Err':>10s} {'Max Err':>10s} {'Est Acc Loss':>13s}")
    print(f"  {'-'*65}")

    for dtype in ["fp16", "int8", "int4"]:
        result = quant_sim.quantize(sample_weights, dtype)
        compressed_size = original_size_mb / result["compression_ratio"]
        print(f"  {result['dtype']:<8s} {result['bits']:>5d} {result['compression_ratio']:>8.0f}x "
              f"{compressed_size:>7.2f} {result['mean_quantization_error']:>10.6f} "
              f"{result['max_quantization_error']:>10.6f} {result['estimated_accuracy_loss']:>12.3f}")

    # 3. Combined benchmark
    print("\n3. Combined Performance Benchmark")
    print("-" * 40)

    scenarios = [
        {"name": "Baseline (FP32, no batch)", "latency_ms": 45, "throughput": 22,
         "memory_mb": 400, "accuracy": 0.9500},
        {"name": "FP16 + Batch=8", "latency_ms": 18, "throughput": 180,
         "memory_mb": 200, "accuracy": 0.9495},
        {"name": "INT8 + Batch=16", "latency_ms": 8, "throughput": 500,
         "memory_mb": 100, "accuracy": 0.9450},
        {"name": "INT8 + Batch=32 + TensorRT", "latency_ms": 4, "throughput": 1200,
         "memory_mb": 95, "accuracy": 0.9440},
    ]

    print(f"  {'Scenario':<35s} {'Latency':>9s} {'Throughput':>11s} "
          f"{'Memory':>8s} {'Accuracy':>9s}")
    print(f"  {'-'*75}")
    for s in scenarios:
        print(f"  {s['name']:<35s} {s['latency_ms']:>7d}ms "
              f"{s['throughput']:>7d} img/s {s['memory_mb']:>6d}MB "
              f"{s['accuracy']:>8.4f}")

    print(f"\n  Recommendation: INT8 + Batch=16 offers the best accuracy/throughput tradeoff")
    print(f"  - 22x throughput improvement over baseline")
    print(f"  - Only 0.5% accuracy loss")
    print(f"  - 4x memory reduction")

    return scenarios


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: TorchServe Deployment")
    print("=" * 60)
    exercise_1_torchserve_handler()

    print("\n\n")
    print("Exercise 2: ONNX Conversion")
    print("=" * 60)
    exercise_2_onnx_conversion()

    print("\n\n")
    print("Exercise 3: Performance Optimization")
    print("=" * 60)
    exercise_3_performance_optimization()
