# 12. Streams and Async

**Previous**: [Profiling with NCU and NSYS](./11_Profiling_with_NCU_NSYS.md) | **Next**: [CUDA Graphs](./13_CUDA_Graphs.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how CUDA streams enable concurrent kernel execution and data transfer
2. Use `cudaMemcpyAsync` with pinned memory for asynchronous transfers
3. Implement a double-buffering pipeline to overlap compute with transfer
4. Measure overlap efficiency with CUDA events and nsys
5. Apply stream-based concurrency to real workloads

---

## 1. The Default Stream Problem

All CUDA operations (kernels + memcpy) in the **default stream** execute **sequentially**:

```
Default stream timeline:
─────────────────────────────────────────────────────────
H→D transfer │  Kernel A  │  Kernel B  │  D→H transfer
─────────────────────────────────────────────────────────
GPU copy engine:  ████                        ████
GPU compute:              ████████

Total time = sum of all operations
```

The GPU has two DMA engines (copy H→D and copy D→H) that can run **simultaneously with** the compute engine. The default stream wastes this parallelism.

---

## 2. CUDA Streams

A stream is an **ordered queue of operations** that execute in-order within the stream, but **different streams can overlap**.

```c
// Create streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
```

### Stream Priorities

Streams can be assigned a priority so the GPU scheduler favors high-priority work:

```c
// Query the valid priority range (lower integer = higher priority)
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
// Typical range: leastPriority=0, greatestPriority=-1

// High-priority stream for real-time inference
cudaStream_t hi_stream;
cudaStreamCreateWithPriority(&hi_stream, cudaStreamNonBlocking, greatestPriority);

// Default-priority stream for background batch processing
cudaStream_t lo_stream;
cudaStreamCreateWithPriority(&lo_stream, cudaStreamNonBlocking, leastPriority);

// High-priority kernel preempts the low-priority one at the next scheduling point
inference_kernel<<<grid, block, 0, hi_stream>>>(d_query);
batch_kernel    <<<grid, block, 0, lo_stream>>> (d_batch);
```

Stream priorities are a scheduling hint, not a hard guarantee. They take effect when two streams compete for the same SM; the GPU may still run both concurrently if resources allow.

```c
// Original stream creation (non-priority variant):

// Operations in stream1 execute concurrently with stream2
kernel_A<<<grid, block, 0, stream1>>>(d_a);      // runs on stream1
cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream2);

// Wait for a specific stream
cudaStreamSynchronize(stream1);  // wait for stream1 only

// Wait for all streams
cudaDeviceSynchronize();

// Destroy
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

---

## 3. Async Memcpy Requires Pinned Memory

`cudaMemcpyAsync` returns immediately to the CPU but only when using **pinned (page-locked) host memory**:

```c
// Pinned allocation
float *h_pinned;
cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault);

// Async copy — CPU continues immediately
cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream);

// CPU can do other work here
do_cpu_work();

// Ensure copy is done before using d_data
cudaStreamSynchronize(stream);
```

With **pageable** memory, `cudaMemcpyAsync` blocks until the data is staged to a temporary pinned buffer — losing the async benefit.

---

## 4. Double-Buffering Pipeline

The classic pattern for overlapping compute with transfer:

```
Without pipelining:
  H→D [chunk 0] │ kernel[0] │ H→D [chunk 1] │ kernel[1] │ D→H [0] │ D→H [1]

With double-buffering (2 streams):
  Stream 0: H→D [chunk 0] │              │ kernel[0] │              │ D→H [0]
  Stream 1:               │ H→D [chunk 1] │           │ kernel[1]    │          │ D→H [1]
  Timeline:  ─────────────┬──────────────┬───────────┬──────────────┬──────────
                          overlap!        overlap!
```

Implementation:

```c
const int NUM_STREAMS = 2;
const int CHUNK = N / NUM_STREAMS;  // elements per chunk

cudaStream_t streams[NUM_STREAMS];
float *d_in[NUM_STREAMS], *d_out[NUM_STREAMS];
float *h_in_pinned, *h_out_pinned;

// Setup
cudaHostAlloc(&h_in_pinned,  N * sizeof(float), cudaHostAllocDefault);
cudaHostAlloc(&h_out_pinned, N * sizeof(float), cudaHostAllocDefault);

for (int s = 0; s < NUM_STREAMS; s++) {
    cudaStreamCreate(&streams[s]);
    cudaMalloc(&d_in[s],  CHUNK * sizeof(float));
    cudaMalloc(&d_out[s], CHUNK * sizeof(float));
}

// Initialize h_in_pinned with data...

// Pipeline: for each chunk, issue H→D, kernel, D→H on its stream
for (int s = 0; s < NUM_STREAMS; s++) {
    int offset = s * CHUNK;

    // H→D: async copy (returns immediately)
    cudaMemcpyAsync(d_in[s], h_in_pinned + offset,
                    CHUNK * sizeof(float), cudaMemcpyHostToDevice, streams[s]);

    // Kernel (runs after H→D on same stream)
    int gridSize = (CHUNK + 255) / 256;
    process_kernel<<<gridSize, 256, 0, streams[s]>>>(d_in[s], d_out[s], CHUNK);

    // D→H: async copy (runs after kernel on same stream)
    cudaMemcpyAsync(h_out_pinned + offset, d_out[s],
                    CHUNK * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
}

// Wait for all streams
cudaDeviceSynchronize();
```

---

## 5. More Than 2 Streams

With N chunks and N streams, the pipeline becomes more efficient. Using 4+ streams can hide the pipeline startup/drain cost:

```c
const int NSTREAMS = 4;
const int CHUNK = (N + NSTREAMS - 1) / NSTREAMS;

cudaStream_t streams[NSTREAMS];
float *d_buf[NSTREAMS];

for (int s = 0; s < NSTREAMS; s++) {
    cudaStreamCreate(&streams[s]);
    cudaMalloc(&d_buf[s], CHUNK * sizeof(float));
}

for (int s = 0; s < NSTREAMS; s++) {
    int sz = min(CHUNK, N - s * CHUNK);
    if (sz <= 0) continue;
    cudaMemcpyAsync(d_buf[s], h_pin + s * CHUNK, sz * sizeof(float),
                    cudaMemcpyHostToDevice, streams[s]);
    process<<<(sz+255)/256, 256, 0, streams[s]>>>(d_buf[s], sz);
    cudaMemcpyAsync(h_out + s * CHUNK, d_buf[s], sz * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[s]);
}
cudaDeviceSynchronize();
```

---

## 6. Stream Events for Synchronization

Events allow one stream to wait for a specific point in another stream:

```c
cudaEvent_t event;
cudaEventCreate(&event);

// Stream 1 records an event when it finishes an operation
kernel_A<<<grid, block, 0, stream1>>>(d_a);
cudaEventRecord(event, stream1);  // record event in stream1's timeline

// Stream 2 waits for stream1's event before proceeding
cudaStreamWaitEvent(stream2, event, 0);  // stream2 inserts a dependency
kernel_B<<<grid, block, 0, stream2>>>(d_b);  // runs AFTER kernel_A in stream1

cudaEventDestroy(event);
```

Use cases:
- Producer/consumer pattern between streams
- Fan-out: one stream's result feeds multiple consumer streams
- Barrier: multiple streams must complete before a final operation

---

## 7. Timing with Events

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Time a single kernel
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);  // wait for stop event

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel: %.3f ms\n", ms);

// Time the entire pipeline (including transfers)
cudaEventRecord(pipeline_start, 0);  // stream 0 = default stream
// ... pipeline operations on other streams ...
cudaDeviceSynchronize();
cudaEventRecord(pipeline_stop, 0);
cudaEventSynchronize(pipeline_stop);
cudaEventElapsedTime(&total_ms, pipeline_start, pipeline_stop);
```

---

## 8. Expected Speedup Analysis

```
Processing N=1GB of data:
  Kernel compute time:   500 ms
  H→D transfer time:     200 ms  (PCIe 4.0 ×16)
  D→H transfer time:     200 ms

Without streams (sequential):
  Total = 200 + 500 + 200 = 900 ms

With 2-stream double buffering:
  Stream 0: H→D(0) → kernel(0) → D→H(0)
  Stream 1:        H→D(1) → kernel(1) → D→H(1)

  Timeline: 100 + 500 + 100 = 700 ms  (transfers overlap with compute)
  Speedup: 900/700 = 1.29×

With many-stream pipelining (transfer << compute):
  Effective: max(500, 200+200) = 500 ms  (nearly perfect overlap)
  Theoretical speedup: 900/500 = 1.8×

Note: speedup depends on transfer_time / compute_time ratio.
If compute >> transfer, pipelining achieves near-2× speedup.
If transfer >> compute, pipelining barely helps.
```

---

## 9. Verifying Overlap with nsys

```bash
nsys profile --output=streams_test ./my_streamed_app
nsys-ui streams_test.nsys-rep
```

In the nsys timeline GUI, look for:
- **CUDA HW** row: shows actual GPU kernel execution
- **MemCpy HtoD** / **MemCpy DtoH** rows: shows DMA transfers
- **Overlap**: transfers and kernels should run simultaneously in different rows

If they appear sequential despite using streams, common causes:
1. Non-pinned host memory (use `cudaHostAlloc`)
2. Default stream operations between streams (blocks all streams)
3. Block size too large — kernel takes entire GPU, no room for overlap
4. Host blocking call between stream launches

---

## Key Takeaways

- CUDA streams enable concurrent execution of kernels and data transfers
- `cudaMemcpyAsync` requires **pinned memory** (`cudaHostAlloc`) to be truly async
- **Double-buffering** overlaps H→D transfer of chunk N+1 with kernel on chunk N
- `cudaStreamWaitEvent` creates dependencies between streams without blocking the host
- Verify overlap with **nsys timeline** — the GPU has separate DMA and compute engines
- Speedup from pipelining = `total_time / max(compute_time, transfer_time)` (ideal case)

---

**Next**: [13. CUDA Graphs](./13_CUDA_Graphs.md) — Capture a sequence of GPU operations as a graph, replay it without CPU launch overhead, and dramatically reduce latency for small-batch inference.
