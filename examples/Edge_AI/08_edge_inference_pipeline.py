"""
08. Edge Inference Pipeline

Demonstrates a complete edge inference pipeline with timing breakdown:
load model, preprocess input, run inference, postprocess output.

Covers:
- Model loading and initialization
- Image preprocessing pipeline
- Inference with timing instrumentation
- Postprocessing (top-k, NMS for detection)
- End-to-end latency breakdown
- Batch processing pipeline

Requirements:
    pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import os
import tempfile
from collections import OrderedDict
from contextlib import contextmanager

print("=" * 60)
print("Edge AI — Edge Inference Pipeline")
print("=" * 60)


# ============================================
# 1. Timing Utilities
# ============================================
print("\n[1] Timing Instrumentation")
print("-" * 40)


class PipelineTimer:
    """Tracks timing for each stage of the inference pipeline."""

    def __init__(self):
        self.stages = OrderedDict()
        self._current_stage = None
        self._start_time = None

    @contextmanager
    def stage(self, name):
        """Context manager for timing a pipeline stage."""
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000  # ms
        if name not in self.stages:
            self.stages[name] = []
        self.stages[name].append(elapsed)

    def summary(self):
        """Print timing summary."""
        print(f"\n  {'Stage':<25} {'Mean (ms)':>10} {'Min':>10} {'Max':>10} {'% Total':>10}")
        print("  " + "-" * 67)

        total_mean = sum(
            sum(times) / len(times) for times in self.stages.values()
        )

        for name, times in self.stages.items():
            mean = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)
            pct = (mean / total_mean) * 100 if total_mean > 0 else 0
            print(f"  {name:<25} {mean:>10.3f} {min_t:>10.3f} {max_t:>10.3f} {pct:>9.1f}%")

        print("  " + "-" * 67)
        print(f"  {'TOTAL':<25} {total_mean:>10.3f}")
        return total_mean

    def reset(self):
        self.stages.clear()


timer = PipelineTimer()
print("PipelineTimer ready — tracks latency per pipeline stage.")


# ============================================
# 2. Model Definition and Loading
# ============================================
print("\n[2] Model Loading")
print("-" * 40)


class EdgeClassifier(nn.Module):
    """Lightweight classifier optimized for edge deployment."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Depthwise separable convolutions for efficiency
        self.features = nn.Sequential(
            # Standard conv for first layer
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),

            # Depthwise separable block 1
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2),

            # Depthwise separable block 2
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# Simulate loading a pre-trained model
model_path = os.path.join(tempfile.gettempdir(), "edge_model.pt")

# Save model
model = EdgeClassifier(num_classes=10)
model.eval()
torch.save(model.state_dict(), model_path)

# Time the loading
with timer.stage("model_load"):
    loaded_model = EdgeClassifier(num_classes=10)
    loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
    loaded_model.eval()

model_size_kb = os.path.getsize(model_path) / 1024
print(f"Model size: {model_size_kb:.1f} KB")
print(f"Parameters: {sum(p.numel() for p in loaded_model.parameters()):,}")
print(f"Load time: {timer.stages['model_load'][0]:.3f} ms")


# ============================================
# 3. Preprocessing Pipeline
# ============================================
print("\n[3] Image Preprocessing")
print("-" * 40)

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),           # Resize to model input size
    transforms.Normalize(                   # ImageNet-style normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def preprocess_image(image_tensor):
    """Preprocess a single image tensor for inference."""
    # In a real pipeline, this would also handle:
    # - Image decoding (JPEG/PNG)
    # - Color space conversion (BGR -> RGB)
    # - Resize with proper interpolation

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dim

    # Apply transforms
    processed = preprocess(image_tensor)
    return processed


# Simulate raw camera input
raw_image = torch.rand(3, 64, 64)  # Random 64x64 RGB image

with timer.stage("preprocess"):
    input_tensor = preprocess_image(raw_image)

print(f"Raw image shape:    {raw_image.shape}")
print(f"Processed shape:    {input_tensor.shape}")
print(f"Preprocess time:    {timer.stages['preprocess'][0]:.3f} ms")


# ============================================
# 4. Inference
# ============================================
print("\n[4] Model Inference")
print("-" * 40)

with timer.stage("inference"):
    with torch.no_grad():
        logits = loaded_model(input_tensor)

print(f"Output logits shape: {logits.shape}")
print(f"Inference time:      {timer.stages['inference'][0]:.3f} ms")


# ============================================
# 5. Postprocessing
# ============================================
print("\n[5] Postprocessing — Top-K Predictions")
print("-" * 40)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def postprocess_classification(logits, class_names, top_k=5):
    """Convert logits to top-k predictions with confidence scores."""
    probs = F.softmax(logits, dim=1)
    top_probs, top_indices = probs.topk(top_k, dim=1)

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append({
            "class": class_names[idx.item()],
            "confidence": prob.item(),
            "class_id": idx.item(),
        })
    return results


with timer.stage("postprocess"):
    predictions = postprocess_classification(logits, CLASS_NAMES, top_k=5)

print("Top-5 predictions:")
for i, pred in enumerate(predictions):
    bar = "#" * int(pred["confidence"] * 40)
    print(f"  {i+1}. {pred['class']:<12} {pred['confidence']:.4f} {bar}")


# ============================================
# 6. End-to-End Pipeline
# ============================================
print("\n[6] End-to-End Pipeline (Multiple Runs)")
print("-" * 40)

timer.reset()

n_runs = 100
for _ in range(n_runs):
    raw = torch.rand(3, 64, 64)

    with timer.stage("preprocess"):
        inp = preprocess_image(raw)

    with timer.stage("inference"):
        with torch.no_grad():
            out = loaded_model(inp)

    with timer.stage("postprocess"):
        preds = postprocess_classification(out, CLASS_NAMES, top_k=3)

total_latency = timer.summary()
print(f"\n  End-to-end latency: {total_latency:.3f} ms")
print(f"  Throughput: {1000 / total_latency:.1f} FPS")


# ============================================
# 7. Batch Processing Pipeline
# ============================================
print("\n[7] Batch Processing Pipeline")
print("-" * 40)

batch_timer = PipelineTimer()

for batch_size in [1, 4, 8, 16]:
    batch_timer.reset()

    for _ in range(50):
        # Simulate batch of images
        raw_batch = torch.rand(batch_size, 3, 64, 64)

        with batch_timer.stage("preprocess"):
            processed_batch = preprocess(raw_batch)

        with batch_timer.stage("inference"):
            with torch.no_grad():
                batch_logits = loaded_model(processed_batch)

        with batch_timer.stage("postprocess"):
            batch_probs = F.softmax(batch_logits, dim=1)
            batch_preds = batch_probs.argmax(dim=1)

    total = sum(
        sum(t) / len(t) for t in batch_timer.stages.values()
    )
    per_sample = total / batch_size
    throughput = 1000 / per_sample

    print(f"  Batch={batch_size:<3}  Total={total:.2f}ms  "
          f"Per-sample={per_sample:.2f}ms  Throughput={throughput:.0f} FPS")


# ============================================
# 8. Pipeline Summary
# ============================================
print("\n[8] Edge Inference Pipeline Summary")
print("-" * 40)

print("""
Typical edge inference pipeline stages:

  1. CAPTURE    — Camera/sensor data acquisition
  2. DECODE     — JPEG/H.264 decode (often hardware-accelerated)
  3. PREPROCESS — Resize, normalize, convert to tensor
  4. INFERENCE  — Model forward pass (the main compute)
  5. POSTPROCESS — Softmax, NMS, thresholding
  6. ACTION     — Send result, trigger actuator, display

Optimization strategies by stage:
  - Preprocess: Use hardware resize (GPU/DSP), batch transforms
  - Inference:  Quantization, pruning, model architecture design
  - Postprocess: Vectorized operations, avoid Python loops
  - Overall:    Pipeline parallelism (overlap stages across frames)
""")

# Cleanup
if os.path.exists(model_path):
    os.remove(model_path)

print("Key takeaways:")
print("- Profile EVERY stage, not just inference (I/O can dominate)")
print("- Use percentiles (P95/P99) for real-time latency requirements")
print("- Batching improves throughput but increases per-frame latency")
print("- Pipeline parallelism can hide preprocess/postprocess time")
