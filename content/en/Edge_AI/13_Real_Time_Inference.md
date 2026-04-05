# 13. Real-Time Inference

**Previous**: [On-Device Training](./12_On_Device_Training.md) | **Next**: [Edge AI for Computer Vision](./14_Edge_AI_for_Computer_Vision.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Optimize inference throughput using dynamic batching and request pipelining
2. Manage memory efficiently on edge devices with pre-allocation and memory pools
3. Apply operator fusion to reduce kernel launch overhead and memory traffic
4. Implement model caching strategies for multi-model edge applications
5. Configure threading and inference scheduling for latency-sensitive workloads
6. Profile inference latency end-to-end and identify bottlenecks

---

Getting a model to run on an edge device is only half the battle. Making it run fast enough for real-time applications -- 30 FPS video processing, sub-20ms voice command recognition, or 100Hz sensor fusion -- requires systematic inference optimization. The difference between a naive deployment and an optimized one can easily be 5-10x in throughput on the same hardware. This lesson covers the techniques that close that gap: batching, pipelining, memory management, operator fusion, caching, and profiling.

---

## 1. Dynamic Batching

### 1.1 Why Batching Matters on Edge

```
+-----------------------------------------------------------------+
|             Single vs Batched Inference                           |
+-----------------------------------------------------------------+
|                                                                   |
|   Single Inference (batch=1):                                    |
|   +------+  +------+  +------+  +------+                        |
|   | Req1 |  | Req2 |  | Req3 |  | Req4 |   Total: 4 * T_single |
|   +------+  +------+  +------+  +------+                        |
|   <--T-->   <--T-->   <--T-->   <--T-->                          |
|                                                                   |
|   Batched Inference (batch=4):                                   |
|   +----------------------------+                                 |
|   | Req1 + Req2 + Req3 + Req4 |            Total: ~1.5 * T      |
|   +----------------------------+                                 |
|   <---------~1.5T------------>                                   |
|                                                                   |
|   Hardware utilization:                                          |
|   - GPU/NPU have parallel compute units                         |
|   - Batch=1 leaves most units idle                               |
|   - Larger batches improve utilization (up to a point)           |
|   - Trade-off: higher throughput vs higher per-request latency   |
|                                                                   |
+-----------------------------------------------------------------+
```

### 1.2 Dynamic Batch Scheduler

```python
#!/usr/bin/env python3
"""Dynamic batching for edge inference server."""

import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class InferenceRequest:
    """A single inference request with its input and callback."""
    request_id: int
    input_data: np.ndarray
    timestamp: float = field(default_factory=time.time)
    result: Optional[np.ndarray] = None
    event: threading.Event = field(default_factory=threading.Event)


class DynamicBatcher:
    """Collects individual requests and batches them for efficient inference.

    The batcher waits up to max_wait_ms for requests to accumulate,
    then runs inference on the entire batch at once. This amortizes
    fixed per-inference overhead (kernel launch, memory transfer)
    across multiple requests.

    Parameters:
        model_fn: Callable that takes a batched numpy array and returns predictions
        max_batch_size: Maximum requests per batch
        max_wait_ms: Maximum time to wait for a batch to fill
    """

    def __init__(self, model_fn: Callable,
                 max_batch_size: int = 8,
                 max_wait_ms: float = 10.0):
        self.model_fn = model_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self.queue = deque()
        self.lock = threading.Lock()
        self.stats = {"batches": 0, "requests": 0, "total_wait_ms": 0}

        # Start batch processing thread
        self.running = True
        self.thread = threading.Thread(target=self._batch_loop, daemon=True)
        self.thread.start()

    def submit(self, input_data: np.ndarray, timeout: float = 1.0) -> np.ndarray:
        """Submit an inference request and wait for result."""
        request = InferenceRequest(
            request_id=id(input_data),
            input_data=input_data
        )

        with self.lock:
            self.queue.append(request)

        # Wait for batch processing to complete this request
        request.event.wait(timeout=timeout)
        return request.result

    def _batch_loop(self):
        """Background thread that processes batched requests."""
        while self.running:
            batch = self._collect_batch()

            if not batch:
                time.sleep(0.001)  # 1ms polling interval
                continue

            # Stack inputs into a single batch
            batch_input = np.stack([r.input_data for r in batch])

            # Run batched inference
            start = time.perf_counter()
            batch_output = self.model_fn(batch_input)
            inference_ms = (time.perf_counter() - start) * 1000

            # Distribute results
            for i, request in enumerate(batch):
                request.result = batch_output[i]
                wait_ms = (time.time() - request.timestamp) * 1000
                self.stats["total_wait_ms"] += wait_ms
                request.event.set()

            self.stats["batches"] += 1
            self.stats["requests"] += len(batch)

    def _collect_batch(self) -> list:
        """Collect up to max_batch_size requests or wait max_wait_ms."""
        batch = []
        deadline = time.time() + self.max_wait_ms / 1000

        while len(batch) < self.max_batch_size and time.time() < deadline:
            with self.lock:
                if self.queue:
                    batch.append(self.queue.popleft())

            if not batch:
                time.sleep(0.0005)

        return batch

    def get_stats(self) -> dict:
        """Return batching statistics."""
        if self.stats["batches"] == 0:
            return self.stats
        return {
            **self.stats,
            "avg_batch_size": self.stats["requests"] / self.stats["batches"],
            "avg_wait_ms": self.stats["total_wait_ms"] / self.stats["requests"],
        }

    def shutdown(self):
        """Stop the batch processing thread."""
        self.running = False
        self.thread.join(timeout=2.0)


# Demonstration
if __name__ == "__main__":
    # Simulate a model
    def mock_model(batch: np.ndarray) -> np.ndarray:
        time.sleep(0.005)  # 5ms inference regardless of batch size
        return np.random.rand(len(batch), 10)

    batcher = DynamicBatcher(mock_model, max_batch_size=8, max_wait_ms=10)

    # Simulate concurrent requests
    results = []
    threads = []
    for i in range(20):
        inp = np.random.rand(224, 224, 3).astype(np.float32)
        t = threading.Thread(target=lambda x=inp: results.append(batcher.submit(x)))
        threads.append(t)
        t.start()
        time.sleep(0.002)  # Stagger requests

    for t in threads:
        t.join()

    print(f"Processed {len(results)} requests")
    print(f"Stats: {batcher.get_stats()}")
    batcher.shutdown()
```

---

## 2. Pipelining

### 2.1 Inference Pipeline Stages

```
+-----------------------------------------------------------------+
|            Three-Stage Inference Pipeline                         |
+-----------------------------------------------------------------+
|                                                                   |
|   Without pipelining (sequential):                               |
|   Frame 1: [Preprocess][  Inference  ][Postprocess]              |
|   Frame 2:                            [Preprocess][Inference]... |
|                                                                   |
|   With pipelining (overlapped):                                  |
|   Stage:     Preprocess    Inference    Postprocess              |
|   Frame 1:   [  Pre  ]                                          |
|   Frame 2:   [  Pre  ]   [  Infer  ]                            |
|   Frame 3:   [  Pre  ]   [  Infer  ]   [  Post  ]              |
|   Frame 4:   [  Pre  ]   [  Infer  ]   [  Post  ]              |
|                                                                   |
|   After the pipeline fills, one frame completes per stage time.  |
|   Throughput = 1 / max(T_pre, T_infer, T_post)                  |
|                                                                   |
+-----------------------------------------------------------------+
```

### 2.2 Pipeline Implementation

```python
#!/usr/bin/env python3
"""Three-stage inference pipeline with producer-consumer queues."""

import numpy as np
import time
import threading
from queue import Queue
from typing import Callable, Optional


class InferencePipeline:
    """Pipelined inference: preprocess, model, postprocess run concurrently.

    Each stage runs in its own thread with Queue-based communication.
    While the model processes frame N, preprocessing handles frame N+1
    and postprocessing handles frame N-1 in parallel.
    """

    def __init__(self,
                 preprocess_fn: Callable,
                 model_fn: Callable,
                 postprocess_fn: Callable,
                 queue_size: int = 4):
        self.preprocess_fn = preprocess_fn
        self.model_fn = model_fn
        self.postprocess_fn = postprocess_fn

        # Inter-stage queues
        self.raw_queue = Queue(maxsize=queue_size)
        self.preprocessed_queue = Queue(maxsize=queue_size)
        self.inference_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue(maxsize=queue_size)

        self.running = False
        self.stats = {"frames": 0, "total_time": 0}

    def start(self):
        """Start pipeline threads."""
        self.running = True
        self.threads = [
            threading.Thread(target=self._preprocess_loop, daemon=True),
            threading.Thread(target=self._inference_loop, daemon=True),
            threading.Thread(target=self._postprocess_loop, daemon=True),
        ]
        for t in self.threads:
            t.start()

    def feed(self, frame: np.ndarray) -> None:
        """Submit a frame to the pipeline."""
        self.raw_queue.put((time.perf_counter(), frame))

    def get_result(self, timeout: float = 1.0) -> Optional[dict]:
        """Get the next processed result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Exception:
            return None

    def _preprocess_loop(self):
        while self.running:
            try:
                timestamp, frame = self.raw_queue.get(timeout=0.1)
                processed = self.preprocess_fn(frame)
                self.preprocessed_queue.put((timestamp, processed))
            except Exception:
                continue

    def _inference_loop(self):
        while self.running:
            try:
                timestamp, data = self.preprocessed_queue.get(timeout=0.1)
                output = self.model_fn(data)
                self.inference_queue.put((timestamp, output))
            except Exception:
                continue

    def _postprocess_loop(self):
        while self.running:
            try:
                timestamp, output = self.inference_queue.get(timeout=0.1)
                result = self.postprocess_fn(output)

                latency = (time.perf_counter() - timestamp) * 1000
                self.stats["frames"] += 1
                self.stats["total_time"] += latency

                self.result_queue.put({
                    "result": result,
                    "latency_ms": latency,
                    "frame_num": self.stats["frames"]
                })
            except Exception:
                continue

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        for t in self.threads:
            t.join(timeout=2.0)

    def get_stats(self) -> dict:
        if self.stats["frames"] == 0:
            return self.stats
        return {
            **self.stats,
            "avg_latency_ms": self.stats["total_time"] / self.stats["frames"],
            "throughput_fps": self.stats["frames"] / (self.stats["total_time"] / 1000),
        }


# Demonstration
if __name__ == "__main__":
    def preprocess(frame):
        """Resize and normalize."""
        time.sleep(0.003)  # 3ms
        return frame.astype(np.float32) / 255.0

    def model_infer(data):
        """Run model."""
        time.sleep(0.015)  # 15ms
        return np.random.rand(10)

    def postprocess(output):
        """Decode predictions."""
        time.sleep(0.002)  # 2ms
        return {"class": int(output.argmax()), "score": float(output.max())}

    pipeline = InferencePipeline(preprocess, model_infer, postprocess)
    pipeline.start()

    # Feed 50 frames
    for i in range(50):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pipeline.feed(frame)
        time.sleep(0.01)  # 100 FPS input rate

    # Collect results
    time.sleep(1.0)
    count = 0
    while True:
        result = pipeline.get_result(timeout=0.1)
        if result is None:
            break
        count += 1

    print(f"Processed {count} frames")
    print(f"Stats: {pipeline.get_stats()}")
    pipeline.stop()
```

---

## 3. Memory Management

### 3.1 Edge Memory Challenges

```
+-----------------------------------------------------------------+
|          Memory Hierarchy on Edge Devices                         |
+-----------------------------------------------------------------+
|                                                                   |
|   Raspberry Pi 4:  1-8 GB RAM (shared with OS and GPU)          |
|   Jetson Nano:     4 GB unified memory (CPU + GPU shared)        |
|   Coral Dev Board: 1 GB RAM + 8 MB Edge TPU SRAM               |
|   MCU (STM32):     256 KB - 1 MB SRAM total                     |
|                                                                   |
|   Common pitfalls:                                               |
|   - Python garbage collector causes latency spikes               |
|   - Repeated numpy allocations fragment memory                   |
|   - Loading multiple models exhausts RAM                         |
|   - Image decoding creates temporary large buffers               |
|                                                                   |
+-----------------------------------------------------------------+
```

### 3.2 Pre-Allocated Buffer Pool

```python
#!/usr/bin/env python3
"""Pre-allocated memory buffers to avoid runtime allocation."""

import numpy as np
from collections import deque
import threading


class BufferPool:
    """Fixed-size pool of pre-allocated numpy arrays.

    Allocating memory during inference causes:
    1. Latency spikes from malloc/free system calls
    2. Memory fragmentation over time
    3. GC pauses when Python reclaims objects

    Pre-allocating a pool eliminates these issues.
    The pool returns buffers instantly and reclaims them
    without invoking the allocator.
    """

    def __init__(self, buffer_shape: tuple, dtype=np.float32,
                 pool_size: int = 8):
        self.buffer_shape = buffer_shape
        self.dtype = dtype
        self.pool_size = pool_size

        # Pre-allocate all buffers upfront
        self.available = deque()
        self.lock = threading.Lock()

        for _ in range(pool_size):
            buf = np.empty(buffer_shape, dtype=dtype)
            self.available.append(buf)

        total_mb = pool_size * np.prod(buffer_shape) * np.dtype(dtype).itemsize / (1024**2)
        print(f"BufferPool: {pool_size} x {buffer_shape} = {total_mb:.1f} MB")

    def acquire(self) -> np.ndarray:
        """Get a buffer from the pool (blocking if none available)."""
        with self.lock:
            if self.available:
                return self.available.popleft()

        # Pool exhausted -- block or raise
        raise RuntimeError(
            f"BufferPool exhausted (pool_size={self.pool_size}). "
            "Increase pool_size or ensure buffers are released."
        )

    def release(self, buffer: np.ndarray) -> None:
        """Return a buffer to the pool."""
        # Zero the buffer to prevent data leakage between frames
        buffer[:] = 0
        with self.lock:
            self.available.append(buffer)

    @property
    def available_count(self) -> int:
        return len(self.available)


class ManagedInference:
    """Inference engine with managed memory buffers."""

    def __init__(self, model_fn, input_shape: tuple, output_shape: tuple,
                 pool_size: int = 4):
        self.model_fn = model_fn

        # Pre-allocate input and output buffers
        self.input_pool = BufferPool(input_shape, np.float32, pool_size)
        self.output_pool = BufferPool(output_shape, np.float32, pool_size)

    def predict(self, raw_data: np.ndarray) -> np.ndarray:
        """Run inference using pooled buffers (zero allocation)."""
        input_buf = self.input_pool.acquire()
        output_buf = self.output_pool.acquire()

        try:
            # Copy data into pre-allocated input buffer
            np.copyto(input_buf, raw_data)

            # Run model (writes to output buffer)
            result = self.model_fn(input_buf)
            np.copyto(output_buf, result)

            return output_buf.copy()  # Return a copy; release buffer
        finally:
            self.input_pool.release(input_buf)
            self.output_pool.release(output_buf)


if __name__ == "__main__":
    def mock_model(x):
        return np.random.rand(1, 10).astype(np.float32)

    engine = ManagedInference(
        mock_model,
        input_shape=(1, 3, 224, 224),
        output_shape=(1, 10),
        pool_size=4
    )

    for i in range(100):
        inp = np.random.rand(1, 3, 224, 224).astype(np.float32)
        out = engine.predict(inp)

    print(f"Input pool available: {engine.input_pool.available_count}")
    print(f"Output pool available: {engine.output_pool.available_count}")
```

---

## 4. Operator Fusion

### 4.1 What is Operator Fusion?

```
+-----------------------------------------------------------------+
|                    Operator Fusion                                |
+-----------------------------------------------------------------+
|                                                                   |
|   Without fusion:                                                |
|   +------+     +------+     +--------+                           |
|   | Conv |---->| BN   |---->| ReLU   |   3 kernel launches      |
|   +------+     +------+     +--------+   3 memory round-trips   |
|      |             |             |                                |
|     Read          Read          Read     (from global memory)    |
|     Compute       Compute       Compute                          |
|     Write         Write         Write    (to global memory)      |
|                                                                   |
|   With fusion (Conv+BN+ReLU):                                   |
|   +--------------------+                                         |
|   | Conv + BN + ReLU   |            1 kernel launch              |
|   +--------------------+            1 memory round-trip          |
|      |                                                           |
|     Read once                                                    |
|     Compute all three                                            |
|     Write once                                                   |
|                                                                   |
|   Benefit: fewer memory accesses dominate latency on edge        |
|   devices where memory bandwidth is the bottleneck.              |
|                                                                   |
+-----------------------------------------------------------------+
```

### 4.2 Manual Operator Fusion in PyTorch

```python
#!/usr/bin/env python3
"""Manual and automatic operator fusion for inference."""

import torch
import torch.nn as nn
import time


class UnfusedBlock(nn.Module):
    """Standard block with separate Conv, BN, ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse Conv2d + BatchNorm2d into a single Conv2d.

    At inference time, BatchNorm is a linear operation:
        y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta

    This can be folded into the convolution weights:
        W_fused = gamma / sqrt(var + eps) * W_conv
        b_fused = gamma * (b_conv - mean) / sqrt(var + eps) + beta

    The fused conv produces identical output with one fewer operation.
    """
    assert not bn.training, "Fuse only in eval mode"

    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        conv.kernel_size, conv.stride, conv.padding,
        conv.dilation, conv.groups, bias=True
    )

    # Compute fused weights
    w_conv = conv.weight.clone()
    b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)

    fused_conv.weight.data = (w_conv * scale.view(-1, 1, 1, 1))
    fused_conv.bias.data = scale * (b_conv - mean) + beta

    return fused_conv


def fuse_model(model: nn.Module) -> nn.Module:
    """Automatically fuse Conv+BN pairs in a model."""
    # PyTorch provides built-in fusion
    model.eval()
    fused = torch.quantization.fuse_modules(model, [
        ["conv", "bn", "relu"]  # Fuse conv+bn+relu together
    ], inplace=False)
    return fused


def benchmark_fusion():
    """Compare fused vs unfused inference speed."""
    block = UnfusedBlock(64, 64).eval()
    x = torch.randn(1, 64, 56, 56)

    # Unfused
    times_unfused = []
    for _ in range(200):
        start = time.perf_counter()
        with torch.no_grad():
            block(x)
        times_unfused.append((time.perf_counter() - start) * 1000)

    # Fused (manual)
    fused_conv = fuse_conv_bn(block.conv, block.bn)
    relu = nn.ReLU()

    times_fused = []
    for _ in range(200):
        start = time.perf_counter()
        with torch.no_grad():
            relu(fused_conv(x))
        times_fused.append((time.perf_counter() - start) * 1000)

    import numpy as np
    print(f"Unfused: {np.mean(times_unfused):.3f} ms")
    print(f"Fused:   {np.mean(times_fused):.3f} ms")
    print(f"Speedup: {np.mean(times_unfused) / np.mean(times_fused):.2f}x")


if __name__ == "__main__":
    benchmark_fusion()
```

### 4.3 TensorRT Automatic Fusion

```python
#!/usr/bin/env python3
"""TensorRT automatically fuses layers during engine build."""

# TensorRT fusion happens automatically during optimization.
# Common fusions TensorRT performs:
trt_fusions = {
    "Conv + BN + ReLU": "Single CUDA kernel",
    "Conv + Add + ReLU": "Residual block fusion (ResNet)",
    "Conv + Scale": "Batch norm folding",
    "MatMul + Add + Activation": "Fused fully connected",
    "Reduce + Elementwise": "Fused normalization",
    "Multi-Head Attention": "Flash Attention kernel (TRT 9+)",
}

for pattern, result in trt_fusions.items():
    print(f"  {pattern:<35} -> {result}")

# To see what TensorRT fused:
# trtexec --onnx=model.onnx --fp16 --dumpLayerInfo --verbose 2>&1 | grep "Fused"
```

---

## 5. Model Caching

### 5.1 Multi-Model Cache

```python
#!/usr/bin/env python3
"""LRU model cache for multi-model edge applications."""

import time
from collections import OrderedDict
from typing import Callable, Any
import os


class ModelCache:
    """LRU cache for loaded models with memory-aware eviction.

    Many edge applications need multiple models (e.g., a detector +
    classifier + tracker). Loading all models simultaneously may
    exceed device RAM. This cache keeps recently-used models in
    memory and evicts the least-recently-used when memory pressure
    is high.
    """

    def __init__(self, max_memory_mb: float = 500.0):
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.cache = OrderedDict()  # name -> (model, size_bytes, last_used)
        self.current_memory = 0
        self.load_fn_registry = {}

        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def register_loader(self, model_name: str, load_fn: Callable,
                        model_path: str):
        """Register a model loader function."""
        self.load_fn_registry[model_name] = (load_fn, model_path)

    def get_model(self, model_name: str) -> Any:
        """Get a model, loading it if not cached."""
        if model_name in self.cache:
            self.stats["hits"] += 1
            # Move to end (most recently used)
            self.cache.move_to_end(model_name)
            model, size, _ = self.cache[model_name]
            self.cache[model_name] = (model, size, time.time())
            return model

        self.stats["misses"] += 1
        return self._load_and_cache(model_name)

    def _load_and_cache(self, model_name: str) -> Any:
        """Load a model and add it to the cache."""
        if model_name not in self.load_fn_registry:
            raise KeyError(f"No loader registered for '{model_name}'")

        load_fn, model_path = self.load_fn_registry[model_name]
        model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0

        # Evict models if needed to make room
        while (self.current_memory + model_size > self.max_memory_bytes
               and self.cache):
            self._evict_lru()

        # Load model
        start = time.perf_counter()
        model = load_fn(model_path)
        load_time = (time.perf_counter() - start) * 1000

        # Cache it
        self.cache[model_name] = (model, model_size, time.time())
        self.current_memory += model_size

        print(f"Loaded '{model_name}' in {load_time:.0f}ms "
              f"(cache: {self.current_memory / 1024**2:.0f} MB)")

        return model

    def _evict_lru(self):
        """Evict the least recently used model."""
        name, (model, size, _) = self.cache.popitem(last=False)
        self.current_memory -= size
        self.stats["evictions"] += 1
        print(f"Evicted '{name}' ({size / 1024**2:.1f} MB)")
        del model

    def get_stats(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cached_models": list(self.cache.keys()),
            "memory_usage_mb": self.current_memory / 1024**2,
        }


if __name__ == "__main__":
    cache = ModelCache(max_memory_mb=100)

    # Register model loaders
    def load_mock(path):
        time.sleep(0.1)  # Simulate loading time
        return f"model_from_{path}"

    cache.register_loader("detector", load_mock, "detector.tflite")
    cache.register_loader("classifier", load_mock, "classifier.tflite")
    cache.register_loader("tracker", load_mock, "tracker.tflite")

    # Simulate model access patterns
    for model_name in ["detector", "classifier", "detector", "tracker",
                       "detector", "classifier"]:
        model = cache.get_model(model_name)

    print(f"\nCache stats: {cache.get_stats()}")
```

---

## 6. Threading and Inference Scheduling

### 6.1 Thread Configuration

```python
#!/usr/bin/env python3
"""Thread configuration for optimal edge inference."""

import os
import numpy as np
import time


def configure_threading(framework: str = "tflite",
                        num_threads: int = None):
    """Configure threading for different inference frameworks.

    Edge devices have limited cores. Over-subscribing threads
    causes contention and cache thrashing. Under-subscribing
    leaves performance on the table.

    Rules of thumb:
    - TFLite: num_threads = num_big_cores (not all cores on big.LITTLE)
    - PyTorch: torch.set_num_threads = physical cores
    - ONNX Runtime: intra_op = physical cores, inter_op = 1
    """
    physical_cores = os.cpu_count()

    if num_threads is None:
        # Default: use physical cores (not hyperthreads)
        # On ARM big.LITTLE, prefer big cores only
        num_threads = max(1, physical_cores // 2)

    if framework == "tflite":
        try:
            from tflite_runtime.interpreter import Interpreter
            interp = Interpreter(
                model_path="model.tflite",
                num_threads=num_threads
            )
            print(f"TFLite: {num_threads} threads")
        except Exception:
            pass

    elif framework == "pytorch":
        import torch
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1)  # One thread for inter-op
        print(f"PyTorch: {num_threads} intra-op, 1 inter-op")

    elif framework == "onnxruntime":
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        print(f"ONNX Runtime: {num_threads} intra-op, 1 inter-op")

    return num_threads


def find_optimal_threads(model_fn, input_data: np.ndarray,
                         max_threads: int = None) -> int:
    """Empirically find the best thread count."""
    if max_threads is None:
        max_threads = os.cpu_count() or 4

    results = {}
    for n in range(1, max_threads + 1):
        # Configure threads (framework-specific)
        os.environ["OMP_NUM_THREADS"] = str(n)
        os.environ["MKL_NUM_THREADS"] = str(n)

        times = []
        for _ in range(50):
            start = time.perf_counter()
            model_fn(input_data)
            times.append((time.perf_counter() - start) * 1000)

        mean_ms = np.mean(times)
        results[n] = mean_ms
        print(f"  {n} threads: {mean_ms:.2f} ms")

    best = min(results, key=results.get)
    print(f"\nOptimal: {best} threads ({results[best]:.2f} ms)")
    return best
```

### 6.2 Priority-Based Inference Scheduler

```python
#!/usr/bin/env python3
"""Priority scheduler for multiple inference tasks."""

import heapq
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass(order=True)
class ScheduledTask:
    priority: int
    deadline_ms: float
    task_fn: Callable = field(compare=False)
    task_id: str = field(compare=False)
    result: Any = field(default=None, compare=False)
    done_event: threading.Event = field(
        default_factory=threading.Event, compare=False
    )


class InferenceScheduler:
    """Priority-based scheduler for latency-sensitive edge inference.

    Different inference tasks have different latency requirements:
    - Safety-critical (e.g., obstacle detection): highest priority
    - Real-time (e.g., pose estimation): medium priority
    - Background (e.g., analytics): lowest priority

    The scheduler ensures high-priority tasks preempt lower ones.
    """

    PRIORITY_CRITICAL = 0  # Safety (must complete by deadline)
    PRIORITY_REALTIME = 1  # Real-time (best effort low-latency)
    PRIORITY_BACKGROUND = 2  # Non-urgent analytics

    def __init__(self, num_workers: int = 2):
        self.task_queue = []
        self.lock = threading.Lock()
        self.running = True
        self.stats = {"completed": 0, "deadline_missed": 0}

        self.workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)

    def schedule(self, task_fn: Callable, priority: int,
                 deadline_ms: float = float("inf"),
                 task_id: str = "") -> ScheduledTask:
        """Schedule an inference task with priority."""
        task = ScheduledTask(
            priority=priority,
            deadline_ms=deadline_ms,
            task_fn=task_fn,
            task_id=task_id
        )

        with self.lock:
            heapq.heappush(self.task_queue, task)

        return task

    def _worker_loop(self):
        while self.running:
            task = None
            with self.lock:
                if self.task_queue:
                    task = heapq.heappop(self.task_queue)

            if task is None:
                time.sleep(0.001)
                continue

            start = time.perf_counter()
            try:
                task.result = task.task_fn()
            except Exception as e:
                task.result = f"Error: {e}"

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats["completed"] += 1

            if elapsed_ms > task.deadline_ms:
                self.stats["deadline_missed"] += 1

            task.done_event.set()

    def wait_for(self, task: ScheduledTask, timeout: float = 5.0) -> Any:
        """Wait for a scheduled task to complete."""
        task.done_event.wait(timeout=timeout)
        return task.result

    def shutdown(self):
        self.running = False
        for t in self.workers:
            t.join(timeout=2.0)


if __name__ == "__main__":
    scheduler = InferenceScheduler(num_workers=2)

    def detect_obstacle():
        time.sleep(0.010)
        return {"obstacle": True, "distance_m": 2.5}

    def estimate_pose():
        time.sleep(0.025)
        return {"keypoints": 17}

    def run_analytics():
        time.sleep(0.050)
        return {"people_count": 3}

    # Schedule with different priorities
    t1 = scheduler.schedule(
        detect_obstacle,
        InferenceScheduler.PRIORITY_CRITICAL,
        deadline_ms=20,
        task_id="obstacle"
    )
    t2 = scheduler.schedule(
        estimate_pose,
        InferenceScheduler.PRIORITY_REALTIME,
        deadline_ms=50,
        task_id="pose"
    )
    t3 = scheduler.schedule(
        run_analytics,
        InferenceScheduler.PRIORITY_BACKGROUND,
        deadline_ms=200,
        task_id="analytics"
    )

    for task in [t1, t2, t3]:
        result = scheduler.wait_for(task)
        print(f"[{task.task_id}] priority={task.priority}: {result}")

    print(f"Stats: {scheduler.stats}")
    scheduler.shutdown()
```

---

## 7. Latency Profiling

### 7.1 End-to-End Profiler

```python
#!/usr/bin/env python3
"""End-to-end inference latency profiler."""

import time
import numpy as np
from contextlib import contextmanager
from collections import defaultdict


class LatencyProfiler:
    """Hierarchical latency profiler for inference pipelines.

    Usage:
        profiler = LatencyProfiler()
        with profiler.measure("preprocess"):
            preprocess(image)
        with profiler.measure("inference"):
            model(tensor)
        with profiler.measure("postprocess"):
            decode(output)
        profiler.report()
    """

    def __init__(self):
        self.timings = defaultdict(list)
        self.current_frame_start = None

    @contextmanager
    def measure(self, stage_name: str):
        """Context manager to measure a pipeline stage."""
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.timings[stage_name].append(elapsed_ms)

    def frame_start(self):
        """Mark the start of a new frame."""
        self.current_frame_start = time.perf_counter()

    def frame_end(self):
        """Mark the end of a frame (records total frame time)."""
        if self.current_frame_start:
            elapsed = (time.perf_counter() - self.current_frame_start) * 1000
            self.timings["_total_frame"].append(elapsed)

    def report(self) -> str:
        """Generate a formatted profiling report."""
        lines = []
        lines.append("\n" + "=" * 65)
        lines.append("  INFERENCE LATENCY PROFILE")
        lines.append("=" * 65)
        lines.append(
            f"  {'Stage':<20} {'Mean':>8} {'P50':>8} {'P95':>8} "
            f"{'P99':>8} {'Count':>6}"
        )
        lines.append("-" * 65)

        total_mean = 0
        for stage, times in sorted(self.timings.items()):
            if stage.startswith("_"):
                continue
            arr = np.array(times)
            mean = np.mean(arr)
            total_mean += mean
            lines.append(
                f"  {stage:<20} {mean:>7.2f}ms "
                f"{np.percentile(arr, 50):>7.2f}ms "
                f"{np.percentile(arr, 95):>7.2f}ms "
                f"{np.percentile(arr, 99):>7.2f}ms "
                f"{len(arr):>6}"
            )

        lines.append("-" * 65)
        lines.append(f"  {'TOTAL (sum)':<20} {total_mean:>7.2f}ms")

        if "_total_frame" in self.timings:
            frame_times = np.array(self.timings["_total_frame"])
            fps = 1000 / np.mean(frame_times)
            lines.append(f"  {'TOTAL (e2e)':<20} {np.mean(frame_times):>7.2f}ms")
            lines.append(f"  {'Throughput':<20} {fps:>7.1f} FPS")

        lines.append("=" * 65)

        report_str = "\n".join(lines)
        print(report_str)
        return report_str

    def reset(self):
        """Clear all collected timings."""
        self.timings.clear()


if __name__ == "__main__":
    profiler = LatencyProfiler()

    for _ in range(100):
        profiler.frame_start()

        with profiler.measure("decode_image"):
            time.sleep(0.002)

        with profiler.measure("preprocess"):
            time.sleep(0.003)

        with profiler.measure("inference"):
            time.sleep(0.015)

        with profiler.measure("postprocess"):
            time.sleep(0.002)

        with profiler.measure("encode_result"):
            time.sleep(0.001)

        profiler.frame_end()

    profiler.report()
```

---

## Practice Exercises

### Exercise 1: Dynamic Batching
1. Implement the `DynamicBatcher` and simulate 100 concurrent requests arriving at random intervals
2. Measure average latency and throughput at batch sizes 1, 4, 8, and 16
3. Plot the throughput vs latency trade-off

### Exercise 2: Memory Optimization
1. Profile a TFLite model's memory usage with `tracemalloc`
2. Implement the `BufferPool` and compare peak memory with and without pooling
3. Measure the effect on GC pause frequency

### Exercise 3: End-to-End Profiling
1. Build a camera inference pipeline (capture -> resize -> normalize -> infer -> decode)
2. Use `LatencyProfiler` to identify the slowest stage
3. Optimize the bottleneck and re-profile to verify improvement

---

**Previous**: [On-Device Training](./12_On_Device_Training.md) | **Next**: [Edge AI for Computer Vision](./14_Edge_AI_for_Computer_Vision.md)
