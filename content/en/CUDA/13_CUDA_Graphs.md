# 13. CUDA Graphs

**Previous**: [Streams and Async](./12_Streams_and_Async.md) | **Next**: [Parallel Reduction](./14_Parallel_Reduction.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why repeated kernel launches have CPU-side overhead
2. Capture a CUDA graph via stream capture
3. Instantiate and replay a graph with `cudaGraphLaunch`
4. Update graph node parameters without re-capturing
5. Apply graphs to small-batch inference to reduce latency

---

## 1. The CPU Launch Overhead Problem

Every CUDA kernel launch has CPU-side overhead:

```
Typical kernel launch overhead:
  cudaLaunchKernel() call:   5–20 μs  (software overhead)
  GPU launch latency:        ~5 μs

For a neural network inference forward pass with 100 kernels:
  Launch overhead alone: 100 × 15 μs = 1,500 μs = 1.5 ms

If the actual GPU compute is 0.5 ms (small batch), the overhead
is 3× the compute time — completely unacceptable for real-time inference.
```

This is why CUDA Graphs were introduced in CUDA 10.0.

---

## 2. CUDA Graph Concept

A **CUDA graph** captures a DAG (Directed Acyclic Graph) of GPU operations:

```
Graph structure:
  MemcpyHtoD(A) ──→ kernel_1(A) ──→ kernel_2(A,B) ──→ MemcpyDtoH(result)
  MemcpyHtoD(B) ──↗
```

Once captured, the **entire graph** can be submitted to the GPU in a **single CPU call** — eliminating per-operation launch overhead. The GPU executes the graph's operations in dependency order.

---

## 3. Stream Capture

The easiest way to create a graph — record normal stream operations:

```c
// Step 1: Begin capture
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Step 2: Issue operations normally — they are CAPTURED, not executed
cudaMemcpyAsync(d_A, h_A, bytes_A, cudaMemcpyHostToDevice, stream);
cudaMemcpyAsync(d_B, h_B, bytes_B, cudaMemcpyHostToDevice, stream);
kernel_1<<<grid, block, 0, stream>>>(d_A, d_tmp);
kernel_2<<<grid, block, 0, stream>>>(d_tmp, d_B, d_out);
cudaMemcpyAsync(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost, stream);

// Step 3: End capture — creates the graph
cudaStreamEndCapture(stream, &graph);

// Step 4: Instantiate (compile the graph for this GPU)
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Step 5: Graph is ready. The stream is no longer capturing.
```

**Important**: During capture, no GPU work actually executes. All operations are recorded into the graph structure.

---

## 4. Replaying the Graph

```c
// Each launch: single CPU call, replays the entire captured sequence
for (int step = 0; step < 1000; step++) {
    // Update input data on host
    prepare_next_batch(h_A, h_B, step);

    // Launch entire graph
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // Read results
    process_output(h_out);
}
```

**Benchmark comparison** (100-kernel neural network forward pass, batch=1):

```
Without graphs: 100 × 15 μs overhead + 0.5 ms compute = 2.0 ms per inference
With graphs:    ~10 μs overhead + 0.5 ms compute       = 0.51 ms per inference
Speedup: 3.9×  (almost entirely from eliminating launch overhead)
```

---

## 5. Updating Graph Parameters

If only the **data pointers or values** change between invocations (not the graph structure), you can update the graph without re-capturing:

```c
// Update a kernel's parameters
cudaKernelNodeParams params;
cudaGraphKernelNodeGetParams(kernelNode, &params);

// Change an argument
float new_alpha = 2.0f;
params.kernelParams[0] = &new_alpha;
cudaGraphKernelNodeSetParams(kernelNode, &params);

// Re-instantiate (much faster than re-capture)
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &params);
```

For pointer updates (e.g., different output buffer):

```c
cudaGraphExecMemcpyNodeSetParams1D(graphExec, memcpyNode,
    d_new_out, h_out, bytes, cudaMemcpyDeviceToHost);
```

---

## 6. Multi-Stream Graph Capture

Capture operations from multiple streams to express parallelism in the graph:

```c
// Two parallel branches
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Capture both streams together
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

// Branch 1 (stream1)
kernel_branch_A<<<grid, block, 0, stream1>>>(d_a);

// Branch 2 (stream2) — captured because of global mode
kernel_branch_B<<<grid, block, 0, stream2>>>(d_b);

// Join: stream2 waits for stream1 (creates dependency in graph)
cudaEvent_t joinEvent;
cudaEventCreate(&joinEvent);
cudaEventRecord(joinEvent, stream1);
cudaStreamWaitEvent(stream2, joinEvent, 0);

// Continuation on stream2 after join
kernel_merge<<<grid, block, 0, stream2>>>(d_a, d_b, d_out);

// End capture — stream1 is the root
cudaStreamEndCapture(stream1, &graph);
```

The resulting graph captures the parallelism between branch A and branch B, allowing the GPU to execute them simultaneously.

---

## 7. Graph Execution with Conditional Paths (CUDA 12.4+)

CUDA 12.4 introduced conditional graph nodes — if-else and while-loop constructs entirely on the GPU, avoiding CPU-GPU round trips:

```c
// (Simplified pseudo-code — actual API uses cudaGraphConditionalHandle)
// Create a conditional node
cudaGraphNode_t condNode;
cudaConditionalNodeParams condParams = { .type = cudaGraphCondTypeIf };
cudaGraphAddConditionalNode(&condNode, graph, deps, ndeps, &condParams);

// Within the conditional, capture "true" and "false" branches as subgraphs
// GPU evaluates the condition at runtime without CPU involvement
```

This eliminates the latency of returning results to the CPU just to decide which GPU operation to run next.

---

## 8. When to Use CUDA Graphs

**Use graphs when:**
- Kernel is very short (< 1 ms) and launch overhead is significant
- The same sequence of operations is repeated many times (inference serving)
- You have many small kernels that can't be fused
- Real-time constraints (robotics, trading, interactive applications)

**Don't use graphs when:**
- The graph structure changes based on data (dynamic control flow)
- Operations are long enough that launch overhead is negligible (> 10 ms)
- Memory allocation patterns change between iterations

```
Heuristic: use graphs if launch_overhead > 10% of compute_time
  compute < 100 μs with 10 kernels: overhead = 150 μs → 150% → definitely use graphs
  compute > 10 ms with 10 kernels:  overhead = 150 μs → 1.5% → skip graphs
```

---

## 9. Complete Example: Inference Serving

```c
// Build the graph once at startup
void setup_inference_graph(Model *m, cudaGraphExec_t *exec) {
    cudaStream_t captureStream;
    cudaStreamCreate(&captureStream);

    cudaGraph_t graph;
    cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal);

    // Forward pass kernels
    embedding_lookup<<<grid, block, 0, captureStream>>>(
        d_input_ids, m->d_embed, m->d_hidden, m->vocab_size, m->d_model);
    for (int layer = 0; layer < m->n_layers; layer++) {
        attention_kernel<<<grid, block, shm, captureStream>>>(
            m->d_hidden, m->d_kv_cache + layer * kv_size,
            m->d_attn_out, m->seq_len, m->d_head);
        ffn_kernel<<<grid, block, 0, captureStream>>>(
            m->d_attn_out, m->d_hidden, m->d_ffn_w1[layer], m->d_ffn_w2[layer]);
    }
    lm_head<<<grid, block, 0, captureStream>>>(m->d_hidden, d_logits, m->vocab_size);

    cudaStreamEndCapture(captureStream, &graph);
    cudaGraphInstantiate(exec, graph, NULL, NULL, 0);

    cudaGraphDestroy(graph);
    cudaStreamDestroy(captureStream);
}

// Inference loop (no graph overhead per call)
void run_inference(cudaGraphExec_t exec, cudaStream_t stream,
                   const int *h_tokens, float *h_logits, int n_tokens) {
    cudaMemcpyAsync(d_input_ids, h_tokens, n_tokens * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaGraphLaunch(exec, stream);
    cudaMemcpyAsync(h_logits, d_logits, vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}
```

---

## Key Takeaways

- CPU kernel launch overhead (5–20 μs per kernel) becomes a bottleneck for short or repeated kernels
- **CUDA Graphs** capture a DAG of GPU operations and replay it with a single CPU call
- **Stream capture** (`cudaStreamBeginCapture/EndCapture`) is the easiest way to build a graph
- `cudaGraphLaunch` replaces all individual kernel launches with one call — typically 20–100× less CPU overhead
- Update node parameters (pointers, scalars) without re-capturing — use `cudaGraphExecKernelNodeSetParams`
- Graphs are most beneficial for inference serving, real-time systems, and any workload with many short kernels

---

**Next**: [14. Parallel Reduction](./14_Parallel_Reduction.md) — Implement the foundational GPU primitive: tree reduction, warp shuffle reduction, multi-stage reduction, and CUB device reduce.
