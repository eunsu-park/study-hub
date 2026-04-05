# 37. Multi-GPU Programming and NCCL

**Previous**: [Fused Kernel Patterns](./36_Fused_Kernel_Patterns.md) | **Next**: [Capstone CUDA Application](./38_Capstone_CUDA_Application.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Transfer data between GPUs using `cudaMemcpyPeer` and enable NVLink P2P access
2. Initialize an NCCL communicator and call `ncclAllReduce` for gradient synchronization
3. Implement data-parallel distributed training with AllReduce gradient averaging
4. Understand tensor parallelism (column/row linear partitioning) as used in Megatron-LM
5. Describe pipeline parallelism (stage assignment, micro-batches) and its trade-offs

---

## 1. Multi-GPU Hardware

```
Interconnect bandwidth comparison:

NVLink (within node):
  NVLink 3.0 (A100):   600 GB/s bidirectional (12 links × 50 GB/s)
  NVLink 4.0 (H100):   900 GB/s bidirectional
  NVSwitch (8-GPU):   full bisection bandwidth between all GPUs

PCIe (within node, no NVLink):
  PCIe 4.0 × 16:      ~32 GB/s (per direction)
  PCIe 5.0 × 16:      ~64 GB/s (per direction)

InfiniBand (across nodes):
  HDR 200 Gb/s:       ~25 GB/s per GPU (with RDMA / GPUDirect)
  NDR 400 Gb/s:       ~50 GB/s per GPU

Rule: gradient AllReduce time ≈ 2 × model_size / bandwidth × (nGPU-1)/nGPU
  7B model (FP16) on 8 A100 (NVLink): 14GB × 7/8 / 600GB/s ≈ 20 ms per iteration
  vs PCIe: 14GB × 7/8 / 32GB/s ≈ 380 ms — 19× slower!
```

---

## 2. Peer-to-Peer Memory Access

```c
// Check and enable P2P access between GPU 0 and GPU 1
void enable_p2p(int src, int dst) {
    int can_p2p;
    cudaDeviceCanAccessPeer(&can_p2p, src, dst);
    if (!can_p2p) {
        fprintf(stderr, "GPU %d cannot directly access GPU %d memory\n", src, dst);
        return;
    }
    cudaSetDevice(src);
    cudaDeviceEnablePeerAccess(dst, 0);
    printf("P2P enabled: GPU %d → GPU %d\n", src, dst);
}

// Direct copy from GPU 1 to GPU 0
void p2p_copy(float *dst_gpu0, const float *src_gpu1, size_t bytes) {
    cudaMemcpyPeer(dst_gpu0, 0,     // destination device
                   src_gpu1, 1,     // source device
                   bytes);
    // With NVLink: copies at ~400 GB/s; without P2P: goes through host memory (~12 GB/s)
}

// Asynchronous P2P copy
void p2p_copy_async(float *dst, int dst_dev,
                    const float *src, int src_dev,
                    size_t bytes, cudaStream_t stream) {
    cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream);
}
```

---

## 3. NCCL Setup and Communicator

NCCL (NVIDIA Collective Communications Library) provides MPI-style collectives optimized for GPU interconnects:

```c
#include <nccl.h>

#define NCCL_CHECK(call) do {                                    \
    ncclResult_t r = call;                                       \
    if (r != ncclSuccess) {                                      \
        fprintf(stderr, "NCCL error %s at %s:%d\n",             \
                ncclGetErrorString(r), __FILE__, __LINE__);      \
        exit(1);                                                 \
    }                                                            \
} while(0)

// Initialize communicator for 4 GPUs (single process)
void init_nccl(ncclComm_t *comms, int nGPU) {
    // Generate unique ID (process 0 generates and broadcasts to all processes)
    ncclUniqueId uid;
    ncclGetUniqueId(&uid);

    // Initialize each communicator (one per GPU)
    for (int g = 0; g < nGPU; g++) {
        cudaSetDevice(g);
        NCCL_CHECK(ncclCommInitRank(&comms[g], nGPU, uid, g));
    }
}

// Multi-process variant: each process calls with its rank
void init_nccl_multiprocess(ncclComm_t *comm, int nranks, int rank,
                             ncclUniqueId uid) {
    NCCL_CHECK(ncclCommInitRank(comm, nranks, uid, rank));
}
```

---

## 4. NCCL AllReduce

AllReduce sums (or max/min/prod) a tensor across all GPUs and distributes the result to all:

```c
// AllReduce: each GPU contributes d_grad, receives the sum / nGPU (mean)
// Called simultaneously on all nGPU devices
void allreduce_gradients(
    float **d_grads,      // d_grads[g] = gradient on GPU g
    int param_count,
    int nGPU,
    ncclComm_t *comms,
    cudaStream_t *streams)
{
    // Launch AllReduce on all GPUs simultaneously (must be in parallel threads or streams)
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < nGPU; g++) {
        cudaSetDevice(g);
        NCCL_CHECK(ncclAllReduce(
            (const void*)d_grads[g],  // send buffer
            (void*)d_grads[g],        // recv buffer (in-place)
            param_count,
            ncclFloat,                // data type
            ncclSum,                  // reduction operation
            comms[g],
            streams[g]));
    }
    NCCL_CHECK(ncclGroupEnd());

    // Scale by 1/nGPU to get mean gradient (done separately per GPU)
    for (int g = 0; g < nGPU; g++) {
        cudaSetDevice(g);
        scale_kernel<<<(param_count+255)/256, 256, 0, streams[g]>>>(
            d_grads[g], 1.f / nGPU, param_count);
    }
}

// Other NCCL collectives:
// ncclBroadcast:   send from root to all GPUs
// ncclAllGather:   each GPU contributes a chunk; all receive the full concatenation
// ncclReduceScatter: reduce + distribute shards (used in ZeRO optimizer)
// ncclSend/ncclRecv: point-to-point (paired with ncclGroupStart/End)
```

---

## 5. Data Parallelism

Each GPU holds a complete model copy, processes a shard of the mini-batch, and synchronizes gradients via AllReduce:

```c
// Data-parallel training loop (4 GPUs, single process using CUDA streams)
void train_data_parallel(
    Model *models,    // models[g] = model on GPU g (identical weights)
    float **d_data,   // d_data[g] = data shard on GPU g
    int nGPU, int steps)
{
    ncclComm_t    comms[4];
    cudaStream_t  streams[4];
    for (int g = 0; g < nGPU; g++) {
        cudaSetDevice(g);
        cudaStreamCreate(&streams[g]);
    }
    init_nccl(comms, nGPU);

    for (int step = 0; step < steps; step++) {
        // --- Forward + backward on each GPU ---
        for (int g = 0; g < nGPU; g++) {
            cudaSetDevice(g);
            // Each GPU processes batch_size/nGPU samples
            forward_backward(models[g], d_data[g], streams[g]);
        }

        // --- Synchronize gradients across all GPUs ---
        NCCL_CHECK(ncclGroupStart());
        for (int g = 0; g < nGPU; g++) {
            cudaSetDevice(g);
            for (int l = 0; l < models[g].n_layers; l++) {
                NCCL_CHECK(ncclAllReduce(
                    models[g].grads[l], models[g].grads[l],
                    models[g].layer_size[l],
                    ncclFloat, ncclSum,
                    comms[g], streams[g]));
            }
        }
        NCCL_CHECK(ncclGroupEnd());

        // --- Optimizer step: each GPU updates its own weights ---
        for (int g = 0; g < nGPU; g++) {
            cudaSetDevice(g);
            optimizer_step(models[g], 1.f/nGPU, streams[g]);  // scale grad by 1/nGPU
        }

        // Weights stay in sync because: identical init + identical AllReduce result
    }
}
```

---

## 6. Tensor Parallelism (Megatron-LM Style)

Tensor parallelism splits individual weight matrices across GPUs. For a linear layer Y = X·W:

```
Column-parallel linear (split W column-wise):
  GPU 0: W_col0 [IC × OC/2]   computes Y0 = X · W_col0
  GPU 1: W_col1 [IC × OC/2]   computes Y1 = X · W_col1
  → AllGather to get full Y = [Y0, Y1]

Row-parallel linear (split W row-wise, follows column-parallel):
  GPU 0: W_row0 [OC/2 × H]   input shard X0, computes partial Z0 = X0 · W_row0
  GPU 1: W_row1 [OC/2 × H]   input shard X1, computes partial Z1 = X1 · W_row1
  → AllReduce (sum) Z = Z0 + Z1

Transformer attention with tensor parallelism (Megatron):
  Each GPU handles H/nGPU attention heads
  No communication needed within attention (heads are independent)
  AllReduce only at the output projection
  Communication per transformer block: 2 AllReduce calls
```

```c
// Column-parallel linear: each GPU has W[:, my_col_start:my_col_end]
void column_parallel_linear(
    cublasHandle_t handle,
    const float *d_X,       // [batch × IC] — same on all GPUs
    const float *d_W_shard, // [IC × local_OC] — different on each GPU
    float *d_Y_shard,       // [batch × local_OC] — local output
    int batch, int IC, int local_OC)
{
    // Standard GEMM (no communication needed for forward pass)
    sgemm_rowmajor(handle, d_X, d_W_shard, d_Y_shard, batch, local_OC, IC);
    // Shards are independently useful (for subsequent row-parallel or AllGather)
}

// AllGather after column-parallel: collect all GPU shards
void allgather_output(
    float *d_Y_shard, int local_OC,
    float *d_Y_full,  int total_OC,
    int batch, int rank, int nGPU,
    ncclComm_t comm, cudaStream_t stream)
{
    NCCL_CHECK(ncclAllGather(
        d_Y_shard,        // send: my shard [batch × local_OC]
        d_Y_full,         // recv: full [batch × total_OC]
        batch * local_OC, // count: elements per GPU
        ncclFloat,
        comm, stream));
}
```

---

## 7. Pipeline Parallelism

Pipeline parallelism splits model layers across GPUs (GPU 0 = layers 0-5, GPU 1 = layers 6-11, etc.):

```
GPipe (naive):
  Stage 0: forward layers 0-5 for micro-batch 0 → send activations to stage 1
  Stage 1: forward layers 6-11 for micro-batch 0 → ...
  Backward in reverse order
  Problem: pipeline bubble = (nStages-1)/nStages of total time wasted

1F1B schedule (PipeDream):
  Interleave forward and backward micro-batches to fill the pipeline bubble
  Each stage alternates: 1 forward step, 1 backward step
  Reduces bubble fraction to 1/m where m = micro-batches per batch
```

```c
// Point-to-point communication for pipeline (send activations to next stage)
void pipeline_send_recv(
    const float *d_act_out, int n_act,  // activations to send
    float *d_act_in,                    // buffer for received activations
    int rank, int nStages,
    ncclComm_t comm, cudaStream_t stream)
{
    NCCL_CHECK(ncclGroupStart());

    // Send to next stage
    if (rank < nStages - 1)
        NCCL_CHECK(ncclSend(d_act_out, n_act, ncclFloat, rank+1, comm, stream));

    // Receive from previous stage
    if (rank > 0)
        NCCL_CHECK(ncclRecv(d_act_in, n_act, ncclFloat, rank-1, comm, stream));

    NCCL_CHECK(ncclGroupEnd());
}
```

---

## 8. Parallelism Comparison

```
Strategy         Comm cost per step     Memory per GPU   Best for
-----------------------------------------------------------------
Data parallel    AllReduce(params)       full model       standard training
Tensor parallel  2×AllReduce per layer   1/nGPU model     large layers, intra-node
Pipeline         send/recv activations   model/nStages    very large models, across nodes

Hybrid (common for large models):
  Data parallel (across nodes) × Tensor parallel (within node) × Pipeline (across nodes)
  Example: GPT-3 training = 64 data parallel × 8 tensor parallel × 8 pipeline parallel
           = 4096 GPUs total

Communication volume comparison (7B FP16 model, 8 GPUs):
  Data parallel:   AllReduce 14GB ≈ 200ms (PCIe) / 20ms (NVLink)
  Tensor parallel: 2 AllReduce per layer × 32 layers
                   each ≈ 4MB → 64 × 4MB = 256MB total ≈ 3ms (NVLink)
  → Tensor parallel wins for intra-node (short, frequent, NVLink)
  → Data parallel wins for inter-node (infrequent, large batch)
```

---

## Key Takeaways

- **NVLink vs PCIe**: NVLink provides 600 GB/s vs ~32 GB/s for PCIe; AllReduce performance difference is ~19× — NVLink is essential for tensor/data parallelism
- **NCCL AllReduce** in-place operation: pass the same pointer as both `sendbuff` and `recvbuff`; always wrap with `ncclGroupStart()`/`ncclGroupEnd()` when launching across multiple GPUs in one process
- **Data parallelism**: simplest approach — each GPU runs the full model on a data shard; gradients synchronized with AllReduce; weights stay identical across GPUs automatically
- **Tensor parallelism**: splits weight matrices column-wise then row-wise across GPUs; only 2 AllReduce calls per transformer layer; requires fast intra-node interconnect (NVLink)
- **Pipeline parallelism**: assigns different layers to different GPUs; uses point-to-point send/recv; 1F1B schedule reduces the pipeline bubble fraction to ~1/m (m = micro-batches)
- For real large-model training, combine all three: data parallel across nodes, tensor parallel within a node, pipeline parallel across groups of nodes

---

**Next**: [38. Capstone CUDA Application](./38_Capstone_CUDA_Application.md) — Integrate everything learned in this course by building either a complete 2D fluid simulation or a small LLM inference engine using custom CUDA kernels.
