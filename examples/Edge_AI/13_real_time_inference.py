"""
13. Real-Time Inference on Edge

Demonstrates techniques for achieving low-latency, real-time inference
on resource-constrained edge devices.

Covers:
- Inference latency profiling and breakdown
- Input preprocessing pipeline optimization
- Batched vs single-sample inference
- Model warmup and JIT compilation effects
- Threading and async inference patterns
- Frame-rate-aware inference scheduling

Requirements:
    pip install torch numpy
"""

import torch
import torch.nn as nn
import time
import threading
import queue
import statistics
from collections import deque
from typing import List

print("=" * 60)
print("Edge AI — Real-Time Inference")
print("=" * 60)


# ============================================
# 1. Define an Edge Inference Model
# ============================================
print("\n[1] Edge Inference Model")
print("-" * 40)


class EdgeClassifier(nn.Module):
    """Lightweight classifier for real-time inference."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = EdgeClassifier()
model.eval()
params = sum(p.numel() for p in model.parameters())
print(f"EdgeClassifier: {params:,} parameters")


# ============================================
# 2. Latency Profiling
# ============================================
print("\n[2] Inference Latency Profiling")
print("-" * 40)


def profile_inference(model, input_tensor, n_warmup=20, n_runs=200):
    """Profile inference latency with detailed statistics."""
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_tensor)

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(input_tensor)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "std_ms": statistics.stdev(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p95_ms": sorted(latencies)[int(0.95 * len(latencies))],
        "p99_ms": sorted(latencies)[int(0.99 * len(latencies))],
    }


resolutions = {
    "64x64": torch.randn(1, 3, 64, 64),
    "128x128": torch.randn(1, 3, 128, 128),
    "224x224": torch.randn(1, 3, 224, 224),
}

print(f"{'Resolution':<12} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Max FPS'}")
print("-" * 60)
for name, tensor in resolutions.items():
    stats = profile_inference(model, tensor)
    max_fps = 1000.0 / stats["mean_ms"]
    print(f"{name:<12} {stats['mean_ms']:<12.2f} {stats['p95_ms']:<12.2f} "
          f"{stats['p99_ms']:<12.2f} {max_fps:.0f}")


# ============================================
# 3. Preprocessing Pipeline
# ============================================
print("\n[3] Preprocessing Pipeline Optimization")
print("-" * 40)


def preprocess_naive(raw_tensor):
    """Naive preprocessing: normalize each image individually."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (raw_tensor - mean) / std
    resized = nn.functional.interpolate(normalized, size=(64, 64), mode="bilinear",
                                        align_corners=False)
    return resized


def preprocess_optimized(raw_tensor, _mean=None, _std=None, _inv_std=None):
    """Optimized: pre-computed constants, multiply instead of divide."""
    if _mean is None:
        _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _inv_std = 1.0 / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    resized = nn.functional.interpolate(raw_tensor, size=(64, 64), mode="bilinear",
                                        align_corners=False)
    return (resized - _mean) * _inv_std


# Cache constants for optimized version
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_INV_STD = 1.0 / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

raw = torch.rand(1, 3, 224, 224)

# Benchmark naive
n_bench = 200
start = time.perf_counter()
for _ in range(n_bench):
    preprocess_naive(raw)
naive_ms = (time.perf_counter() - start) / n_bench * 1000

# Benchmark optimized
start = time.perf_counter()
for _ in range(n_bench):
    preprocess_optimized(raw, _MEAN, None, _INV_STD)
opt_ms = (time.perf_counter() - start) / n_bench * 1000

print(f"Naive preprocessing:     {naive_ms:.3f} ms/frame")
print(f"Optimized preprocessing: {opt_ms:.3f} ms/frame")
print(f"Speedup: {naive_ms / opt_ms:.2f}x")


# ============================================
# 4. Batched vs Single Inference
# ============================================
print("\n[4] Batched vs Single-Sample Inference")
print("-" * 40)

n_frames = 8
single_inputs = [torch.randn(1, 3, 64, 64) for _ in range(n_frames)]
batched_input = torch.randn(n_frames, 3, 64, 64)

# Single-sample inference
start = time.perf_counter()
with torch.no_grad():
    for inp in single_inputs:
        model(inp)
single_total_ms = (time.perf_counter() - start) * 1000

# Batched inference
start = time.perf_counter()
with torch.no_grad():
    model(batched_input)
batched_total_ms = (time.perf_counter() - start) * 1000

print(f"Processing {n_frames} frames:")
print(f"  Single-sample: {single_total_ms:.2f} ms total "
      f"({single_total_ms / n_frames:.2f} ms/frame)")
print(f"  Batched:       {batched_total_ms:.2f} ms total "
      f"({batched_total_ms / n_frames:.2f} ms/frame)")
print(f"  Throughput gain: {single_total_ms / batched_total_ms:.2f}x")


# ============================================
# 5. JIT Compilation Effects
# ============================================
print("\n[5] TorchScript JIT Compilation Effects")
print("-" * 40)

input_64 = torch.randn(1, 3, 64, 64)
traced = torch.jit.trace(model, input_64)

# Compare eager vs JIT
eager_stats = profile_inference(model, input_64)
jit_stats = profile_inference(traced, input_64)

print(f"{'Mode':<15} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Std (ms)'}")
print("-" * 50)
print(f"{'Eager':<15} {eager_stats['mean_ms']:<12.3f} "
      f"{eager_stats['p95_ms']:<12.3f} {eager_stats['std_ms']:.3f}")
print(f"{'JIT Traced':<15} {jit_stats['mean_ms']:<12.3f} "
      f"{jit_stats['p95_ms']:<12.3f} {jit_stats['std_ms']:.3f}")
print(f"JIT speedup: {eager_stats['mean_ms'] / jit_stats['mean_ms']:.2f}x")


# ============================================
# 6. Frame-Rate-Aware Scheduling
# ============================================
print("\n[6] Frame-Rate-Aware Inference Scheduler")
print("-" * 40)
print("Process frames at a target FPS, skip frames if inference is too slow.\n")


class FrameScheduler:
    """Schedule inference to maintain target FPS."""

    def __init__(self, model, target_fps=30):
        self.model = model
        self.target_fps = target_fps
        self.frame_budget_ms = 1000.0 / target_fps
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "latencies": [],
        }

    def process_frame(self, frame_tensor):
        """Process a frame within the time budget."""
        start = time.perf_counter()

        with torch.no_grad():
            result = self.model(frame_tensor)

        latency_ms = (time.perf_counter() - start) * 1000
        self.stats["processed"] += 1
        self.stats["latencies"].append(latency_ms)
        return result, latency_ms

    def should_skip(self, elapsed_since_last_ms):
        """Determine if a frame should be skipped to maintain FPS."""
        if elapsed_since_last_ms < self.frame_budget_ms * 0.5:
            return True  # Too early, skip to save power
        return False

    def simulate_stream(self, n_frames=100, input_shape=(1, 3, 64, 64)):
        """Simulate processing a video stream."""
        last_process_time = time.perf_counter()

        for i in range(n_frames):
            now = time.perf_counter()
            elapsed_ms = (now - last_process_time) * 1000

            if self.should_skip(elapsed_ms):
                self.stats["skipped"] += 1
                continue

            frame = torch.randn(*input_shape)
            _, latency = self.process_frame(frame)
            last_process_time = time.perf_counter()

        return self.stats


scheduler = FrameScheduler(traced, target_fps=30)
stats = scheduler.simulate_stream(n_frames=200)

avg_lat = statistics.mean(stats["latencies"]) if stats["latencies"] else 0
effective_fps = 1000.0 / avg_lat if avg_lat > 0 else 0

print(f"Target FPS: 30")
print(f"Frames received: 200")
print(f"Frames processed: {stats['processed']}")
print(f"Frames skipped:   {stats['skipped']}")
print(f"Avg latency: {avg_lat:.2f} ms")
print(f"Effective FPS: {effective_fps:.0f}")


# ============================================
# 7. Async Inference with Producer-Consumer
# ============================================
print("\n[7] Async Inference (Producer-Consumer Pattern)")
print("-" * 40)
print("Decouple frame capture from inference using a thread-safe queue.\n")


class AsyncInferenceEngine:
    """Async inference using a background thread."""

    def __init__(self, model, max_queue_size=10):
        self.model = model
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.stats = {"processed": 0, "dropped": 0}

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def submit(self, frame_tensor, frame_id):
        """Submit a frame for inference (drops if queue full)."""
        try:
            self.input_queue.put_nowait((frame_id, frame_tensor))
            return True
        except queue.Full:
            self.stats["dropped"] += 1
            return False

    def _inference_loop(self):
        while self.running:
            try:
                frame_id, tensor = self.input_queue.get(timeout=0.1)
                with torch.no_grad():
                    result = self.model(tensor)
                self.result_queue.put((frame_id, result))
                self.stats["processed"] += 1
            except queue.Empty:
                continue

    def get_results(self):
        """Collect all available results."""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results


# Run async inference simulation
engine = AsyncInferenceEngine(traced, max_queue_size=5)
engine.start()

# Simulate rapid frame submission
for i in range(50):
    frame = torch.randn(1, 3, 64, 64)
    engine.submit(frame, frame_id=i)
    time.sleep(0.002)  # Simulate 500 FPS camera capture

time.sleep(0.5)  # Wait for processing
engine.stop()

results = engine.get_results()
print(f"Submitted: 50 frames")
print(f"Processed: {engine.stats['processed']}")
print(f"Dropped (queue full): {engine.stats['dropped']}")
print(f"Results collected: {len(results)}")


# ============================================
# 8. End-to-End Latency Breakdown
# ============================================
print("\n[8] End-to-End Latency Breakdown")
print("-" * 40)

input_raw = torch.rand(1, 3, 224, 224)

# Stage 1: Preprocessing
start = time.perf_counter()
for _ in range(100):
    preprocessed = preprocess_optimized(input_raw, _MEAN, None, _INV_STD)
preproc_ms = (time.perf_counter() - start) / 100 * 1000

# Stage 2: Inference
start = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        output = traced(preprocessed)
infer_ms = (time.perf_counter() - start) / 100 * 1000

# Stage 3: Postprocessing
start = time.perf_counter()
for _ in range(100):
    probs = torch.softmax(output, dim=1)
    class_id = probs.argmax(dim=1).item()
    confidence = probs.max().item()
postproc_ms = (time.perf_counter() - start) / 100 * 1000

total_ms = preproc_ms + infer_ms + postproc_ms

print(f"{'Stage':<20} {'Time (ms)':<12} {'% of Total'}")
print("-" * 44)
print(f"{'Preprocessing':<20} {preproc_ms:<12.3f} {preproc_ms / total_ms * 100:.1f}%")
print(f"{'Inference':<20} {infer_ms:<12.3f} {infer_ms / total_ms * 100:.1f}%")
print(f"{'Postprocessing':<20} {postproc_ms:<12.3f} {postproc_ms / total_ms * 100:.1f}%")
print(f"{'Total':<20} {total_ms:<12.3f} 100.0%")
print(f"\nMax sustainable FPS: {1000.0 / total_ms:.0f}")


# ============================================
# 9. Summary
# ============================================
print("\n[9] Summary")
print("-" * 40)
print("Key takeaways:")
print("- Profile P95/P99 latency, not just mean, for real-time guarantees")
print("- Optimize preprocessing: pre-compute constants, multiply > divide")
print("- Batching improves throughput but increases per-frame latency")
print("- TorchScript JIT typically gives 10-30% inference speedup")
print("- Frame scheduling and async patterns prevent dropped frames")
print("- Measure end-to-end latency: preprocessing can dominate total time")
