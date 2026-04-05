# 06. Warp Execution and Divergence

**Previous**: [Shared Memory and Tiling](./05_Shared_Memory_and_Tiling.md) | **Next**: [Atomic Operations](./07_Atomic_Operations.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain SIMT execution and why warp divergence serializes threads
2. Identify divergence patterns and restructure code to minimize them
3. Use `__ballot_sync`, `__any_sync`, `__all_sync` for warp-level predicates
4. Implement a warp-level reduction with `__shfl_down_sync`
5. Write warp-uniform control flow in performance-critical kernels

---

## 1. SIMT: One Instruction, Multiple Threads

A warp of 32 threads executes as a single unit — all threads execute the **same instruction** each cycle. This is SIMT: Single Instruction, Multiple Threads.

When all 32 threads take the same branch: no performance loss.
When threads diverge: the warp serializes — executes both paths sequentially.

```
Warp of 4 threads (simplified):

Code:
  if (x > 0) {
      y = x * 2;     // Path A
  } else {
      y = -x;        // Path B
  }

Thread 0: x =  5  (takes Path A)
Thread 1: x = -3  (takes Path B)
Thread 2: x =  7  (takes Path A)
Thread 3: x = -1  (takes Path B)

Execution with divergence:
  Cycle 1: Threads 0,2 execute y = x*2;  Threads 1,3 are MASKED (idle)
  Cycle 2: Threads 1,3 execute y = -x;   Threads 0,2 are MASKED (idle)
  Total: 2 cycles instead of 1 → 50% efficiency
```

The SIMT hardware uses an **active mask** — disabled threads do nothing but still wait.

---

## 2. Real-World Divergence Patterns

### Pattern 1: Data-dependent branches (bad)

```c
// Histogram-style: different threads take different paths
if (value > threshold) {
    result = compute_heavy(value);    // only some threads execute
} else {
    result = 0.0f;
}
```

**Fix**: If the branch bodies are cheap, replace with arithmetic:

```c
// No divergence — same instruction for all threads
float mask   = (float)(value > threshold);
result = mask * compute_light(value);  // multiply by 0 instead of skipping
```

This only works if `compute_light` is cheap enough to run on all threads.

### Pattern 2: Thread-ID based divergence (sometimes unavoidable)

```c
// Common in reduction — lower-half threads do work, others idle
if (threadIdx.x < s) {
    smem[threadIdx.x] += smem[threadIdx.x + s];
}
```

When `s = 16`: threads 0–15 execute, 16–31 are masked. Only a 2-way divergence at the warp boundary — relatively harmless for whole-warp reductions.

### Pattern 3: Loop divergence (dangerous)

```c
// Threads iterate different numbers of times
int count = data[threadIdx.x];  // different values per thread!
for (int i = 0; i < count; i++) {
    process(i);
}
```

The warp runs until the **last thread finishes** — all other threads wait, masked. If counts vary widely (e.g., 0 to 100), average utilization could be ~50%.

---

## 3. Warp-Level Intrinsics

Since Volta (CC 7.0), CUDA provides **warp-level primitives** with explicit synchronization masks. The mask specifies which threads participate.

### `__ballot_sync`: which threads satisfy a condition?

```c
unsigned mask = __ballot_sync(0xFFFFFFFF, predicate);
// Returns a 32-bit integer where bit k = 1 if thread k's predicate is true
// 0xFFFFFFFF means "all 32 threads participate"

// Example: count how many threads in the warp have value > 0
unsigned active = __ballot_sync(0xFFFFFFFF, value > 0.0f);
int count = __popc(active);  // popcount = number of set bits
```

### `__any_sync` / `__all_sync`

```c
// True if ANY thread in the mask satisfies the condition
if (__any_sync(0xFFFFFFFF, has_work)) {
    // At least one thread has work to do
}

// True if ALL threads satisfy the condition
if (__all_sync(0xFFFFFFFF, is_valid)) {
    // Safe to proceed — all threads have valid data
}
```

---

## 4. Warp Shuffle: Communication Without Shared Memory

Threads within a warp can directly exchange register values without going through shared memory — `__shfl_sync` family:

```c
// Read a register value from a specific lane in the warp
int   __shfl_sync     (unsigned mask, int var, int srcLane, int width=32);

// Shift down by delta lanes (useful for reduction)
float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width=32);

// Shift up by delta lanes
float __shfl_up_sync  (unsigned mask, float var, unsigned delta, int width=32);

// XOR-based exchange (butterfly pattern)
float __shfl_xor_sync (unsigned mask, float var, int laneMask, int width=32);
```

No `__syncthreads()` needed — all threads in the warp execute synchronously.

### Example: Broadcast lane 0's value to all threads

```c
float leader_val = __shfl_sync(0xFFFFFFFF, my_val, 0);
// Every thread now has the value that thread 0 had
```

---

## 5. Warp Reduction with `__shfl_down_sync`

The classic warp-level reduction — sum all 32 values without shared memory:

```c
__device__ float warp_reduce_sum(float val) {
    // Each step: thread k gets the sum of k and k+offset
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    // Thread 0 now holds the sum of all 32 values
    return val;
}
```

Execution trace for 4 threads (simplified):

```
Initial:  T0=1, T1=2, T2=3, T3=4
delta=2:  T0 += T2 → T0=4,  T1 += T3 → T1=6
delta=1:  T0 += T1 → T0=10, T1=6
Result:   T0 = 10 = 1+2+3+4  ✓
```

**Performance**: 5 shuffle instructions vs 5 shared memory loads+stores. About **2× faster** than the shared memory version and requires no synchronization.

---

## 6. Block-Level Reduction Using Warp Reduction

Combining warp reduction with shared memory for a full block:

```c
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];  // one slot per warp (max 32 warps/block)

    int lane   = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    // Step 1: reduce within each warp
    val = warp_reduce_sum(val);

    // Step 2: first thread in each warp writes to shared memory
    if (lane == 0) warp_sums[warpId] = val;
    __syncthreads();

    // Step 3: first warp reduces the warp sums
    val = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
    if (warpId == 0) val = warp_reduce_sum(val);

    // Thread 0 has the total block sum
    return val;
}
```

---

## 7. Predicated Execution: Avoiding Divergence

On short branches, the compiler can use **predicated execution** — both paths execute, but the result of the non-taken path is discarded:

```ptx
// Assembly equivalent of: if (x > 0) y = 1; else y = -1;
setp.gt.f32  p, x, 0.0
@p  mov.f32  y, 1.0
@!p mov.f32  y, -1.0
```

Both instructions execute (no divergence), but the predicate gate discards the wrong result. This is faster than a true branch for very short bodies. The compiler does this automatically for simple conditionals.

---

## 8. Warp-Level Reduction Benchmark

```c
// Benchmark: shared memory reduction vs shuffle reduction for sum of N floats
// Block size = 256 (8 warps)

// Shared memory version: ~3.2 μs for N=1M
// Shuffle version:       ~1.8 μs for N=1M   (1.8× speedup)

// Profile difference: shuffle avoids ~8 shared memory read-write pairs per block
```

---

## 9. Best Practices Summary

| Pattern | Bad | Good |
|---------|-----|------|
| Short branch | `if (flag) y = a*b; else y = 0;` | `y = flag * a * b;` |
| Warp reduction | shared memory 5-step | `__shfl_down_sync` 5-step |
| Lane-specific work | `if (lane == 0) { ... }` | Acceptable — 1-thread overhead |
| Data-dependent loops | Loop count per-thread | Precompute uniform count |
| Predicate check | `__any_sync` before expensive path | ✓ Use to skip when all idle |

---

## Key Takeaways

- **Divergence serializes warps** — both paths execute, masked threads idle; 2-way divergence = 50% efficiency
- **`__ballot_sync`** converts predicate results to a bitmask for warp-level logic
- **`__shfl_down_sync`** enables register-to-register communication within a warp — faster and simpler than shared memory for reductions
- **Warp reduction** in 5 shuffle steps replaces 5 shared memory operations — ~2× faster
- Short branches with cheap bodies → use arithmetic instead of control flow (no divergence cost)

---

**Next**: [07. Atomic Operations](./07_Atomic_Operations.md) — Implement lock-free counters, histogram kernels, and measure the throughput cost of atomic contention.
