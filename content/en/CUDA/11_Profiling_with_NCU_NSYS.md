# 11. Profiling with NCU and NSYS

**Previous**: [Roofline Model](./10_Roofline_Model.md) | **Next**: [Streams and Async](./12_Streams_and_Async.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use Nsight Compute (`ncu`) to profile individual kernel bottlenecks
2. Use Nsight Systems (`nsys`) to analyze timeline and system-level behavior
3. Identify the top performance bottleneck from metric output
4. Interpret the Speed-of-Light (SOL) sections in the ncu report
5. Build a profiling-first optimization workflow

---

## 1. Two Tools, Two Scales

| Tool | Focus | Granularity |
|------|-------|-------------|
| **ncu** (Nsight Compute) | Individual kernel internals | Per-kernel hardware counters |
| **nsys** (Nsight Systems) | Full application timeline | CPU+GPU+PCIe activity |

**Workflow**: Use `nsys` first to find which kernels are slow. Then use `ncu` on those specific kernels to understand why.

---

## 2. Nsight Systems (`nsys`)

```bash
# Basic timeline capture
nsys profile --stats=true --output=myapp ./my_application

# With GPU metrics included
nsys profile \
    --gpu-metrics-device=all \
    --cudabacktrace=all \
    --output=myapp \
    ./my_application

# View the report
nsys-ui myapp.nsys-rep
```

### Reading nsys stats output

```
[6/7] Executing 'gpukernsum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Name
 --------  ---------------  ---------  --------  --------  ----
    74.3%       145,230,000        100   1452300  1449200   my_gemm_kernel
    15.2%        29,800,000       1000     29800    29500   my_elementwise_kernel
     6.8%        13,300,000        100   133000    132000   cudaMemcpy (HtoD)
```

Key observations from this output:
- `my_gemm_kernel` consumes 74% of GPU time — optimize this first
- `my_elementwise_kernel` runs 1000 times — possible to batch or fuse
- PCIe transfer is relatively small — memory management is fine

---

## 3. Nsight Compute (`ncu`) — Basic Usage

```bash
# Profile a single kernel (all metrics — slow)
ncu --kernel-name myKernel ./my_app

# Profile with specific metric sets (faster)
ncu --set default ./my_app

# Profile and output to file for GUI viewing
ncu --output profile_report ./my_app
ncu-ui profile_report.ncu-rep

# Profile specific metrics
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
    sm__warps_active.avg.pct_of_peak_sustained_active \
    ./my_app
```

---

## 4. Speed-of-Light (SOL) Sections

The most important part of the ncu report is the **Speed of Light** section:

```
Section: Speed Of Light Throughput
─────────────────────────────────────────────────────────
Metric Name                         Metric Value   Unit
─────────────────────────────────────────────────────────
DRAM Frequency                          1,593.0  MHz
SM Frequency                              765.0  MHz
Elapsed Cycles                          2,048,0  cycle
Memory [%]                                 83.2    %   ← memory utilization
DRAM Throughput [%]                        80.1    %   ← % of peak BW
Elapsed Cycles                             2048  cycle
Duration                                   2.68   ms
L1/TEX Cache Throughput [%]                82.0    %
L2 Cache Throughput [%]                    78.3    %
SM Active Cycles                        1,945,0  cycle
Compute (SM) [%]                           12.3    %   ← compute utilization
─────────────────────────────────────────────────────────
```

**Interpretation**:
- `DRAM Throughput = 80%` + `Compute = 12%` → **memory-bound kernel**
- The gap between 80% and 100% represents optimization potential in memory access patterns
- If both were >70%, the kernel would be well-balanced

**Decision tree**:
```
DRAM Throughput > 70% AND Compute < 30% → Memory-bound → fix coalescing, reduce loads
Compute > 70% AND DRAM < 30%            → Compute-bound → vectorize, Tensor Core, unroll
Both > 60%                              → Well-balanced → near-roofline performance
Both < 40%                              → Something else (occupancy, launch overhead, synchronization)
```

---

## 5. Key Metric Groups

### Memory metrics

```bash
ncu --metrics \
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\           # global load bytes
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\           # global store bytes
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\ # sectors/req (coalescing)
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,\         # L2 misses
    dram__bytes_read.sum,\                                   # DRAM reads
    dram__bytes_write.sum                                    # DRAM writes
    ./my_kernel
```

Target values:
- `sectors_per_request` = 1.0 → perfect coalescing
- `l2_miss_rate` < 20% → good L2 cache utilization

### Compute metrics

```bash
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\   # SM utilization %
    sm__warps_active.avg.pct_of_peak_sustained_active,\  # occupancy
    sm__inst_executed_pipe_fma.sum,\                     # FMA instructions
    sm__inst_executed_pipe_alu.sum                       # ALU instructions
    ./my_kernel
```

### Shared memory metrics

```bash
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\  # load bank conflicts
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\  # store bank conflicts
    l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum                # shared load bytes
    ./my_kernel
```

---

## 6. Practical Profiling Workflow

```bash
# Step 1: Find the slow kernel
nsys profile --stats=true ./my_app 2>&1 | grep -A 20 gpukernsum

# Step 2: Get the SOL overview
ncu --set default --kernel-name "slow_kernel" ./my_app

# Step 3: Diagnose the bottleneck
# If memory-bound:
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    --kernel-name "slow_kernel" ./my_app

# If compute-bound:
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name "slow_kernel" ./my_app

# Step 4: Measure specific fix impact
ncu --metrics dram__bytes_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./my_app_before ./my_app_after
```

---

## 7. Interpreting Warp State Distribution

Nsight Compute reports where warps spend their time:

```
Warp State Statistics
─────────────────────────────────────────────────
State              Avg (# warps)   % of Active Cycles
─────────────────────────────────────────────────
Stall MIO Throttle      18.4           52.3%   ← memory issue
Issue                    8.2           23.3%   ← actual work
Stall Wait                4.1           11.7%
Stall Long Scoreboard     3.3            9.4%   ← long latency (global mem)
Other                     1.5            3.3%
─────────────────────────────────────────────────
```

**Stall MIO Throttle** (52%) means: warps are blocked waiting for memory instructions to be issued — memory-bound, possibly due to too many outstanding memory requests or poor coalescing.

**Stall Long Scoreboard** = waiting for global memory loads to return — expected for memory-bound kernels.

---

## 8. NCU for Tensor Cores

For kernels using Tensor Cores (WMMA or cuBLAS):

```bash
ncu --metrics \
    sm__inst_executed_pipe_tensor_op_hmma.sum,\    # FP16 Tensor Core ops
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active \
    ./tensor_core_kernel
```

If `hmma` ops are zero but you expected Tensor Core usage, the kernel is falling back to CUDA cores.

---

## 9. Common Performance Issues and Their ncu Signatures

| Issue | ncu Signature |
|-------|--------------|
| Non-coalesced global memory | `sectors_per_request` >> 1 |
| Shared memory bank conflicts | `bank_conflicts` > 0 |
| Low occupancy | `warps_active` < 50% |
| Register spilling | Check `--metrics launch__registers_per_thread` + ptxas output |
| CPU-GPU serialization | `nsys` timeline shows GPU idle between kernels |
| Kernel too short to hide launch overhead | `Duration` < 10 μs in nsys |
| Tensor Core underuse | `tensor_op_hmma` count is zero |

---

## Key Takeaways

- **nsys** → find which kernels are slow; **ncu** → understand why
- The **Speed-of-Light** section instantly tells you: memory-bound (DRAM% high, Compute% low) or compute-bound (flip)
- `sectors_per_request` is the single most actionable coalescing metric — target 1.0
- Warp state distribution reveals stall reasons: `Long Scoreboard` = global memory, `MIO Throttle` = instruction queue full
- Profile BEFORE optimizing — gut feel is wrong 80% of the time about where bottlenecks are

---

**Next**: [12. Streams and Async](./12_Streams_and_Async.md) — Use CUDA streams to overlap computation with data transfer, implement double-buffering pipelines, and time events precisely.
