# 14. GPU Computing

[← Previous: Particle Systems and Effects](13_Particle_Systems_and_Effects.md) | [Next: Real-Time Rendering Techniques →](15_Real_Time_Rendering_Techniques.md)

---

## Learning Objectives

1. Understand GPU architecture: SIMT execution model, warps/wavefronts, and occupancy
2. Explain compute shaders and their execution model (work groups, shared memory)
3. Implement common GPGPU parallel patterns: reduction, prefix sum, and parallel sorting
4. Apply GPU computing to image processing tasks
5. Compare GPU computing frameworks: CUDA, OpenCL, compute shaders, and WebGPU
6. Understand the GPU memory hierarchy: global, shared, local (registers), and constant memory
7. Recognize practical considerations: CPU-GPU data transfer, synchronization, and occupancy tuning
8. Implement GPU-style parallel algorithms in Python (simulated) and understand their GPU analogues

---

## Why This Matters

Modern GPUs are not just rendering machines -- they are massively parallel processors capable of trillions of operations per second. A high-end GPU has thousands of cores running in lockstep, delivering 10-100x the throughput of a CPU for suitable workloads. This power drives not only real-time graphics but also deep learning, scientific simulation, cryptocurrency mining, and data analytics.

Understanding how GPUs compute is essential for anyone working in graphics or performance-critical computing. Even if you never write CUDA or Vulkan compute shaders directly, knowing what the GPU is good at (and bad at) fundamentally shapes how you design algorithms and data structures.

---

## 1. GPU Architecture

### 1.1 CPU vs. GPU Design Philosophy

The CPU is optimized for **latency** -- it executes a single thread as fast as possible with large caches, branch prediction, and out-of-order execution. The GPU is optimized for **throughput** -- it executes thousands of threads simultaneously, hiding memory latency through massive parallelism.

```
CPU: Few cores, complex control logic
┌─────────────────────────────────┐
│  Core 0  │  Core 1  │  Core 2  │  ...  (4-64 cores)
│  [ALU]   │  [ALU]   │  [ALU]   │
│  [Cache] │  [Cache] │  [Cache] │
│  [Branch Pred]  [OoO Exec]     │
└─────────────────────────────────┘

GPU: Many cores, simple control logic
┌────────────────────────────────────────────────┐
│  SM 0          │  SM 1          │  SM 2    ... │  (30-150 SMs)
│  ┌──┬──┬──┬──┐│  ┌──┬──┬──┬──┐│               │
│  │32│32│32│32││  │32│32│32│32││               │  (32-128 cores/SM)
│  └──┴──┴──┴──┘│  └──┴──┴──┴──┘│               │
│  [Shared Mem]  │  [Shared Mem]  │               │
│  [Scheduler]   │  [Scheduler]   │               │
└────────────────────────────────────────────────┘
```

### 1.2 SIMT: Single Instruction, Multiple Threads

GPUs execute instructions in a **SIMT** (Single Instruction, Multiple Threads) model. A group of threads (called a **warp** on NVIDIA, **wavefront** on AMD) executes the same instruction simultaneously:

- **NVIDIA**: warp = 32 threads
- **AMD**: wavefront = 32 or 64 threads

All threads in a warp execute in lockstep. If threads take different branches (divergence), both paths are executed serially, and threads not on the active path are masked off. This makes branching expensive.

### 1.3 Thread Hierarchy

GPU threads are organized hierarchically:

| Level | NVIDIA Term | OpenGL/Vulkan Term | Size |
|-------|-------------|-------------------|------|
| Single thread | Thread | Invocation | 1 |
| Lock-step group | Warp (32) | Subgroup | 32-64 |
| Cooperative group | Thread Block | Work Group | 64-1024 |
| Full dispatch | Grid | Dispatch | Millions |

**Work group** threads can:
- Synchronize via **barriers** (all threads wait until everyone reaches the barrier)
- Share data through **shared memory** (fast on-chip SRAM)
- Communicate within the group

Threads in different work groups **cannot** directly communicate or synchronize during a dispatch.

### 1.4 Occupancy

**Occupancy** is the ratio of active warps to the maximum number of warps an SM can support. Higher occupancy means better latency hiding:

$$\text{Occupancy} = \frac{\text{Active warps per SM}}{\text{Maximum warps per SM}}$$

Occupancy is limited by:
- **Registers per thread**: More registers → fewer threads fit on the SM
- **Shared memory per work group**: Larger shared memory → fewer groups per SM
- **Work group size**: Must be a multiple of the warp size

**Rule of thumb**: Aim for 50%+ occupancy. Sometimes lower occupancy with better register use is faster (instruction-level parallelism compensates).

---

## 2. Compute Shaders

### 2.1 What Is a Compute Shader?

A **compute shader** is a GPU program that runs outside the graphics pipeline. It reads and writes arbitrary buffers and textures, performing general-purpose computation.

In OpenGL/Vulkan:

```glsl
#version 430
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer InputBuffer {
    float data_in[];
};

layout(std430, binding = 1) buffer OutputBuffer {
    float data_out[];
};

uniform uint N;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= N) return;

    // Each thread processes one element
    data_out[idx] = data_in[idx] * 2.0;
}
```

Dispatch from CPU:
```cpp
glDispatchCompute(ceil(N / 256.0), 1, 1);
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
```

### 2.2 Work Group Layout

Compute shaders define a 3D work group size (local size) and are dispatched as a 3D grid of work groups:

- **1D dispatch** (typical for arrays): `local_size_x = 256`, dispatch `(ceil(N/256), 1, 1)`
- **2D dispatch** (images): `local_size_x = 16, local_size_y = 16`, dispatch `(ceil(W/16), ceil(H/16), 1)`
- **3D dispatch** (volumes): Used for volumetric data

**Choosing work group size**:
- Must be a multiple of the warp size (32 or 64) for full utilization
- Common choices: 64, 128, 256, or 512 threads
- For 2D: 16x16 = 256 threads is a popular choice
- Larger groups use more shared memory but allow more inter-thread communication

### 2.3 Shared Memory

**Shared memory** (called "groupshared" in HLSL, "shared" in GLSL) is fast on-chip memory accessible to all threads in a work group:

```glsl
shared float cache[256];

void main() {
    uint local_idx = gl_LocalInvocationID.x;
    uint global_idx = gl_GlobalInvocationID.x;

    // Load from global memory into shared memory (fast subsequent access)
    cache[local_idx] = data_in[global_idx];

    // Barrier: ensure all threads have loaded before proceeding
    barrier();

    // Now every thread can read any element in cache[]
    // Example: average with neighbors
    float left  = (local_idx > 0) ? cache[local_idx - 1] : cache[local_idx];
    float right = (local_idx < 255) ? cache[local_idx + 1] : cache[local_idx];
    data_out[global_idx] = (left + cache[local_idx] + right) / 3.0;
}
```

Shared memory is typically 16-96 KB per SM, with access latency ~100x faster than global memory.

---

## 3. GPGPU Parallel Patterns

### 3.1 Parallel Reduction

**Problem**: Compute the sum (or max, min, etc.) of $N$ elements.

**Sequential**: $O(N)$

**Parallel**: $O(N/P + \log P)$ where $P$ is the number of threads.

**Algorithm**:

```
Step 0: [5, 3, 8, 1, 4, 7, 2, 6]    (8 elements)
Step 1: [8, _, 9, _, 11, _, 8, _]    (add pairs)
Step 2: [17, _, _, _, 19, _, _, _]   (add pairs of sums)
Step 3: [36, _, _, _, _, _, _, _]    (final sum)
```

Each step halves the active threads. After $\log_2 N$ steps, the result is in element 0.

```python
import numpy as np

def gpu_style_reduction(data, op=np.add):
    """
    Simulate a GPU parallel reduction.
    Each 'step' represents one GPU synchronization barrier.
    The actual GPU executes all active threads simultaneously.
    """
    n = len(data)
    # Why copy: GPU would work on a buffer, not modify the input
    buf = data.copy().astype(float)

    stride = 1
    steps = 0
    while stride < n:
        # In a real GPU, all threads with (idx % (2*stride) == 0) execute
        # Why this stride pattern: avoids bank conflicts in shared memory
        for i in range(0, n, 2 * stride):
            if i + stride < n:
                buf[i] = op(buf[i], buf[i + stride])
        stride *= 2
        steps += 1

    print(f"  Reduction: {n} elements in {steps} steps (log2={np.log2(n):.0f})")
    return buf[0]


data = np.array([5, 3, 8, 1, 4, 7, 2, 6, 9, 10, 11, 12, 13, 14, 15, 16])
total = gpu_style_reduction(data)
print(f"  Sum = {total} (expected: {data.sum()})")
```

### 3.2 Prefix Sum (Scan)

**Problem**: Given array $[a_0, a_1, ..., a_{n-1}]$, compute the running sum $[a_0, a_0+a_1, a_0+a_1+a_2, ...]$ (inclusive scan) or $[0, a_0, a_0+a_1, ...]$ (exclusive scan).

Prefix sum is a fundamental building block used in:
- Stream compaction (removing dead particles)
- Radix sort
- Histogram equalization
- Building spatial data structures

**Blelloch scan** (work-efficient parallel prefix sum):

**Up-sweep** (reduce):
```
Step 0: [3, 1, 7, 0, 4, 1, 6, 3]
Step 1: [3, 4, 7, 7, 4, 5, 6, 9]   (pairs summed)
Step 2: [3, 4, 7, 11, 4, 5, 6, 14]  (quads summed)
Step 3: [3, 4, 7, 11, 4, 5, 6, 25]  (total in last)
```

**Down-sweep** (distribute):
```
Step 0: [3, 4, 7, 11, 4, 5, 6, 0]  (set last to 0)
Step 1: [3, 4, 7, 0, 4, 5, 6, 11]
Step 2: [3, 0, 7, 4, 4, 11, 6, 16]
Step 3: [0, 3, 4, 11, 11, 15, 16, 22]  (exclusive scan)
```

```python
def blelloch_scan(data):
    """
    Work-efficient parallel exclusive prefix sum (Blelloch 1990).
    Two phases: up-sweep (reduce) and down-sweep (distribute).
    Total work: O(n), span: O(log n).
    """
    n = len(data)
    buf = data.copy().astype(float)

    # Up-sweep: build partial sums (like reduction)
    stride = 1
    while stride < n:
        # Why 2*stride-1 indexing: we accumulate at specific positions
        for i in range(2 * stride - 1, n, 2 * stride):
            buf[i] += buf[i - stride]
        stride *= 2

    # Set last element to 0 (exclusive scan identity)
    buf[n - 1] = 0

    # Down-sweep: distribute partial sums
    stride = n // 2
    while stride >= 1:
        for i in range(2 * stride - 1, n, 2 * stride):
            temp = buf[i - stride]
            buf[i - stride] = buf[i]
            buf[i] += temp
        stride //= 2

    return buf


data = np.array([3, 1, 7, 0, 4, 1, 6, 3])
result = blelloch_scan(data)
expected = np.concatenate([[0], np.cumsum(data)[:-1]])
print(f"  Input:    {data}")
print(f"  Scan:     {result.astype(int)}")
print(f"  Expected: {expected}")
```

### 3.3 Parallel Sorting: Bitonic Sort

**Bitonic sort** is a comparison-based sorting algorithm well-suited for GPU execution because its comparison pattern is fixed (data-independent), enabling efficient parallel implementation.

A **bitonic sequence** is a sequence that first monotonically increases, then decreases (or vice versa). Bitonic merge sorts a bitonic sequence in $O(\log n)$ parallel steps.

**Full bitonic sort**: $O(\log^2 n)$ parallel steps, each step comparing and swapping pairs of elements.

```python
def bitonic_sort(data):
    """
    Bitonic sort: GPU-friendly parallel sorting algorithm.
    Comparison pattern is independent of data values,
    making it ideal for SIMT execution (no divergence).
    """
    arr = data.copy()
    n = len(arr)

    # k: size of bitonic subsequences (doubles each outer step)
    k = 2
    while k <= n:
        # j: comparison distance (halves each inner step)
        j = k // 2
        while j >= 1:
            # All comparisons at this (k, j) level can execute in parallel
            for i in range(n):
                partner = i ^ j  # XOR determines comparison partner

                if partner > i:  # Avoid double-processing
                    # Direction: ascending if in first half of k-block, else descending
                    ascending = ((i & k) == 0)

                    if ascending:
                        if arr[i] > arr[partner]:
                            arr[i], arr[partner] = arr[partner], arr[i]
                    else:
                        if arr[i] < arr[partner]:
                            arr[i], arr[partner] = arr[partner], arr[i]

            j //= 2
        k *= 2

    return arr


data = np.array([8, 3, 5, 1, 7, 2, 6, 4])
sorted_data = bitonic_sort(data)
print(f"  Input:  {data}")
print(f"  Sorted: {sorted_data}")
```

### 3.4 Stream Compaction

**Problem**: Given an array and a predicate, extract elements that satisfy the predicate (removing "dead" particles, for example).

**GPU approach**:
1. Evaluate predicate for each element in parallel → produce flag array [1, 0, 1, 1, 0, ...]
2. Prefix sum on flags → compute output index for each surviving element
3. Scatter: each surviving element writes to its computed output index

Total work: $O(n)$, span: $O(\log n)$.

---

## 4. Image Processing on GPU

### 4.1 Convolution

Image convolution applies a kernel (filter) to each pixel:

$$(I * K)(x, y) = \sum_{i=-r}^{r}\sum_{j=-r}^{r} K(i, j) \cdot I(x+i, y+j)$$

This is naturally parallel: each pixel's output depends only on its neighborhood.

**GPU implementation**: Each thread processes one output pixel. Shared memory stores the local tile of the image to avoid redundant global memory reads.

### 4.2 Separable Filters

Many useful kernels (Gaussian, Sobel, box filter) are **separable**: the 2D kernel can be decomposed into two 1D passes:

$$K_{2D} = K_{\text{row}} \cdot K_{\text{col}}^T$$

For a Gaussian with standard deviation $\sigma$:

$$G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-x^2/(2\sigma^2)}$$

**Separable convolution** reduces work from $O(n^2 k^2)$ to $O(n^2 \cdot 2k)$ where $k$ is the kernel radius. For a 5x5 kernel, this is 2.5x faster; for 11x11, it is 5.5x faster.

### 4.3 Python Implementation: Image Convolution

```python
import numpy as np

def gpu_style_convolution(image, kernel):
    """
    Simulate GPU-style image convolution.
    In a real GPU, each thread computes one output pixel.
    Shared memory would cache the local tile.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    h, w = image.shape

    # Why we pad: border pixels need neighbors that may be outside the image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)

    # Each (i, j) represents a GPU thread computing one output pixel
    for i in range(h):
        for j in range(w):
            # In a real GPU compute shader, this inner sum would use
            # shared memory to avoid redundant global memory reads
            val = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    val += padded[i + ki, j + kj] * kernel[ki, kj]
            output[i, j] = val

    return output


def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def separable_gaussian(image, sigma, radius=None):
    """
    Separable Gaussian blur: two 1D passes instead of one 2D pass.
    This is how GPUs actually implement Gaussian blur.
    """
    if radius is None:
        radius = int(3 * sigma)
    size = 2 * radius + 1

    # 1D Gaussian kernel
    ax = np.arange(size) - radius
    k1d = np.exp(-ax**2 / (2 * sigma**2))
    k1d /= k1d.sum()

    h, w = image.shape

    # Horizontal pass: each thread processes one pixel in a row
    padded_h = np.pad(image, ((0, 0), (radius, radius)), mode='reflect')
    temp = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            temp[i, j] = np.dot(padded_h[i, j:j+size], k1d)

    # Vertical pass: each thread processes one pixel in a column
    padded_v = np.pad(temp, ((radius, radius), (0, 0)), mode='reflect')
    output = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            output[i, j] = np.dot(padded_v[i:i+size, j], k1d)

    return output


# Demo: Compare 2D vs separable Gaussian
np.random.seed(42)
image = np.random.rand(64, 64)

kernel_2d = gaussian_kernel(5, 1.0)
result_2d = gpu_style_convolution(image, kernel_2d)
result_sep = separable_gaussian(image, 1.0, radius=2)

diff = np.abs(result_2d - result_sep).max()
print(f"  2D vs Separable max difference: {diff:.10f}")
print(f"  2D kernel operations per pixel: {5*5} = 25")
print(f"  Separable operations per pixel: {5+5} = 10")
```

---

## 5. GPU Computing Frameworks

### 5.1 CUDA (NVIDIA)

**CUDA** (Compute Unified Device Architecture) is NVIDIA's proprietary GPU computing platform (2007). It provides C/C++ extensions for writing GPU kernels.

```cuda
// CUDA kernel: vector addition
__global__ void vecAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch: 256 threads per block
vecAdd<<<ceil(N/256.0), 256>>>(d_A, d_B, d_C, N);
```

**Pros**: Mature ecosystem (cuBLAS, cuDNN, cuFFT), excellent tooling (Nsight), dominant in ML/HPC.
**Cons**: NVIDIA-only, proprietary.

### 5.2 OpenCL

**OpenCL** (Open Computing Language) is a cross-platform standard for parallel computing (CPUs, GPUs, FPGAs, DSPs).

```opencl
__kernel void vecAdd(__global float* A, __global float* B,
                     __global float* C, int N) {
    int idx = get_global_id(0);
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

**Pros**: Cross-vendor (AMD, NVIDIA, Intel, ARM), cross-device.
**Cons**: More verbose API, generally behind CUDA in optimization and tooling.

### 5.3 Graphics API Compute Shaders

**OpenGL Compute Shaders** (4.3+), **Vulkan Compute**, **Metal Compute**, and **DirectX Compute Shaders** provide GPU computing within the graphics API:

| Feature | OpenGL CS | Vulkan CS | Metal CS | DX12 CS |
|---------|-----------|-----------|----------|---------|
| API integration | Good | Excellent | Excellent | Excellent |
| Cross-platform | Linux/Win/macOS* | Linux/Win/Android | macOS/iOS | Windows/Xbox |
| Explicit control | Low | High | High | High |
| Async compute | No | Yes | Yes | Yes |

*OpenGL is deprecated on macOS

### 5.4 WebGPU

**WebGPU** is the next-generation web GPU API, replacing WebGL:

```wgsl
// WebGPU compute shader (WGSL language)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input)) { return; }
    output[idx] = input[idx] * 2.0;
}
```

**Pros**: Web-native, modern explicit API design, runs in browsers.
**Cons**: Still maturing (as of 2025), performance overhead vs. native APIs.

### 5.5 Comparison Summary

| Framework | Vendor Lock | Language | Performance | Ecosystem |
|-----------|-------------|----------|-------------|-----------|
| CUDA | NVIDIA only | C/C++/PTX | Best (NVIDIA) | Largest (ML/HPC) |
| OpenCL | Cross-vendor | C99/SPIR-V | Good | Moderate |
| Vulkan Compute | Cross-vendor | GLSL/SPIR-V | Excellent | Growing |
| Metal Compute | Apple only | MSL | Excellent (Apple) | Apple ecosystem |
| WebGPU | Cross-vendor | WGSL | Good | Web/emerging |

---

## 6. GPU Memory Hierarchy

### 6.1 Memory Types

```
┌──────────────────────────────────────────────┐
│                  Global Memory                │  (VRAM: 8-24 GB, ~500 GB/s)
│  Large capacity, high latency (~400 cycles)   │
├──────────────────────────────────────────────┤
│         L2 Cache (~4-6 MB, shared)           │
├────────────┬────────────┬────────────────────┤
│   SM 0     │   SM 1     │   SM 2    ...      │
│ ┌────────┐ │ ┌────────┐ │                    │
│ │Shared  │ │ │Shared  │ │  (16-96 KB/SM)     │
│ │Memory  │ │ │Memory  │ │  ~20 cycles        │
│ ├────────┤ │ ├────────┤ │                    │
│ │Register│ │ │Register│ │  (256 KB/SM)       │
│ │File    │ │ │File    │ │  1 cycle           │
│ └────────┘ │ └────────┘ │                    │
└────────────┴────────────┴────────────────────┘
```

| Memory | Scope | Size | Latency | Bandwidth |
|--------|-------|------|---------|-----------|
| Registers | Per thread | ~255 regs/thread | 1 cycle | Highest |
| Shared memory | Per work group | 16-96 KB/SM | ~20 cycles | ~10 TB/s |
| L1 cache | Per SM | 32-128 KB | ~30 cycles | High |
| L2 cache | Global | 4-6 MB | ~200 cycles | ~4 TB/s |
| Global (VRAM) | Global | 8-24 GB | ~400 cycles | ~500-900 GB/s |
| Constant | Global (read-only) | 64 KB | ~4 cycles (cached) | Broadcast |

### 6.2 Memory Access Patterns

**Coalesced access**: When consecutive threads access consecutive memory addresses, the GPU combines the reads into a single wide transaction (128 bytes). This is critical for performance:

```
Good (coalesced):
  Thread 0 reads data[0]
  Thread 1 reads data[1]
  Thread 2 reads data[2]
  ...
  → 1 memory transaction

Bad (strided):
  Thread 0 reads data[0]
  Thread 1 reads data[128]
  Thread 2 reads data[256]
  ...
  → 32 separate transactions (32x slower!)
```

**Structure of Arrays (SoA) vs. Array of Structures (AoS)**:

```
AoS (bad for GPU):
  struct Particle { float x, y, z, vx, vy, vz; };
  Particle particles[N];
  // Thread k reads particles[k].x → non-coalesced

SoA (good for GPU):
  float x[N], y[N], z[N], vx[N], vy[N], vz[N];
  // Thread k reads x[k] → coalesced
```

### 6.3 Bank Conflicts

Shared memory is divided into **banks** (typically 32 banks, 4 bytes each). If two threads access the same bank simultaneously, the accesses are serialized (bank conflict).

```
No conflict: Each thread accesses a different bank
  Thread 0 → Bank 0, Thread 1 → Bank 1, ...

Bank conflict (2-way): Two threads hit the same bank
  Thread 0 → Bank 0, Thread 1 → Bank 0  (serialized: 2x slower)

Broadcast: All threads read the SAME address (no conflict)
  Thread 0 → Bank 0[addr X], Thread 1 → Bank 0[addr X]  (broadcast)
```

**Mitigation**: Pad shared memory arrays to avoid stride patterns that hit the same bank.

---

## 7. Practical Considerations

### 7.1 Data Transfer

CPU-GPU data transfer over PCIe is often the bottleneck:

| Bus | Bandwidth | Latency |
|-----|-----------|---------|
| PCIe 3.0 x16 | ~16 GB/s | ~10 us |
| PCIe 4.0 x16 | ~32 GB/s | ~10 us |
| PCIe 5.0 x16 | ~64 GB/s | ~10 us |
| GPU memory (VRAM) | ~500-900 GB/s | ~400 cycles |

**Rule**: Minimize CPU-GPU transfers. Keep data on the GPU as long as possible. Compute on GPU, transfer only final results.

### 7.2 Synchronization

**Within a work group**: Use barriers (`barrier()` in GLSL, `__syncthreads()` in CUDA).

**Between work groups**: Use a memory barrier and relaunch another dispatch. Compute shaders cannot synchronize across work groups within a single dispatch.

**CPU-GPU**: Use fences to wait for GPU completion. Avoid round-trip synchronization in tight loops (pipeline stalls).

### 7.3 When to Use the GPU

GPUs excel when:
- The problem is **massively parallel** (same operation on millions of elements)
- Data is **large enough** to amortize transfer overhead (>100K elements)
- Memory access is **regular** (coalesced, predictable)
- Branching is **minimal** (uniform control flow)

GPUs are poor at:
- Sequential algorithms with data dependencies
- Small problems (GPU launch overhead dominates)
- Irregular memory access (pointer chasing, trees)
- Heavy branching (warp divergence)

---

## 8. Python Simulation of GPU Patterns

```python
import numpy as np
import time

def simulate_parallel_map(data, func):
    """
    Simulate a GPU parallel map: apply func to each element independently.
    On a real GPU, each thread processes one element simultaneously.
    In Python, we use NumPy vectorization as an analogue.
    """
    return func(data)


def simulate_parallel_reduction(data):
    """Simulate GPU reduction using NumPy (which uses SIMD internally)."""
    return np.sum(data)


def simulate_image_convolution_tiled(image, kernel, tile_size=16):
    """
    Simulate tiled GPU convolution with shared memory.
    Each tile loads its data into 'shared memory' (a local array),
    including halo pixels needed for the kernel.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    h, w = image.shape

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)

    # Process in tiles (simulating GPU work groups)
    num_tiles_y = (h + tile_size - 1) // tile_size
    num_tiles_x = (w + tile_size - 1) // tile_size

    tiles_processed = 0
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            y_start = ty * tile_size
            x_start = tx * tile_size
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)

            # Simulated "shared memory" load: tile + halo
            # Why halo: kernel needs neighboring pixels outside the tile
            shared_y_start = y_start
            shared_y_end = y_end + 2 * pad_h
            shared_x_start = x_start
            shared_x_end = x_end + 2 * pad_w

            shared_mem = padded[shared_y_start:shared_y_end,
                                shared_x_start:shared_x_end]

            # Each "thread" in the tile computes one output pixel
            for local_y in range(y_end - y_start):
                for local_x in range(x_end - x_start):
                    val = 0.0
                    for ki in range(kh):
                        for kj in range(kw):
                            val += shared_mem[local_y + ki, local_x + kj] * kernel[ki, kj]
                    output[y_start + local_y, x_start + local_x] = val

            tiles_processed += 1

    print(f"  Processed {tiles_processed} tiles ({tile_size}x{tile_size})")
    return output


# --- Performance comparison demo ---

N = 1_000_000
data = np.random.rand(N)

# Map operation: square each element
start = time.perf_counter()
result_loop = np.array([x ** 2 for x in data])  # "CPU serial"
loop_time = time.perf_counter() - start

start = time.perf_counter()
result_vec = simulate_parallel_map(data, lambda x: x ** 2)  # "GPU parallel"
vec_time = time.perf_counter() - start

print(f"  Map {N:,} elements:")
print(f"    Python loop: {loop_time:.4f}s")
print(f"    NumPy (GPU-like): {vec_time:.4f}s")
print(f"    Speedup: {loop_time/vec_time:.1f}x")

# Reduction
start = time.perf_counter()
total = simulate_parallel_reduction(data)
red_time = time.perf_counter() - start
print(f"\n  Reduction of {N:,} elements: sum={total:.2f} in {red_time:.6f}s")

# Tiled image convolution
print(f"\n  Tiled image convolution:")
image = np.random.rand(128, 128)
kernel = gaussian_kernel(5, 1.0)  # Reuse from previous section
result = simulate_image_convolution_tiled(image, kernel, tile_size=16)
print(f"  Output shape: {result.shape}, mean: {result.mean():.4f}")
```

---

## 9. GPU in Graphics: Beyond Rendering

GPUs in graphics engines serve multiple compute roles beyond traditional rendering:

| Task | Technique |
|------|-----------|
| Physics simulation | Particle systems, cloth, fluids (SPH) |
| Culling | Frustum/occlusion culling on GPU |
| Animation | Skinning, blend shapes on GPU |
| Terrain | Heightmap generation, tessellation |
| Post-processing | Bloom, SSAO, motion blur, tone mapping |
| AI/ML | Neural rendering, DLSS, denoising |
| Ray tracing | BVH traversal, intersection (RT cores) |

**Async compute**: Modern APIs (Vulkan, DX12, Metal) support running compute and graphics work simultaneously on different GPU queues, improving utilization.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| SIMT | Threads in a warp (32) execute the same instruction; divergence is costly |
| Work group | Threads that can synchronize and share memory (64-1024 threads) |
| Occupancy | Ratio of active warps to maximum; higher = better latency hiding |
| Shared memory | Fast on-chip SRAM; ~20 cycles; shared within work group |
| Coalesced access | Consecutive threads reading consecutive addresses = efficient |
| Reduction | Parallel sum in $O(\log n)$ steps; fundamental GPGPU pattern |
| Prefix sum | Running total in $O(\log n)$ span; building block for compaction, sort |
| Bitonic sort | Fixed comparison pattern; $O(\log^2 n)$ parallel steps; no divergence |
| Separable filters | Decompose 2D convolution into two 1D passes; reduces work by $O(k)$ |
| SoA vs AoS | Structure of Arrays is GPU-friendly (coalesced); Array of Structures is not |

## Exercises

1. **Reduction variants**: Implement parallel max and parallel argmax using the GPU-style reduction pattern. Test with an array of 1024 random values.

2. **Prefix sum verification**: Implement the Blelloch scan for a power-of-two-sized array. Verify the result against `np.cumsum`. Extend it to handle non-power-of-two sizes.

3. **Histogram**: Design a GPU algorithm to compute a histogram of pixel intensities (256 bins) for a grayscale image. Explain how you would handle atomic operations for bin increments.

4. **Tiling analysis**: For a 1920x1080 image with a 7x7 Gaussian kernel and work groups of 16x16 threads, calculate: (a) how many tiles, (b) the shared memory per tile (including halo), and (c) the total number of global memory reads saved compared to non-tiled.

5. **SoA transformation**: Given an `AoS` particle structure `[{x,y,z,vx,vy,vz}, ...]`, convert it to `SoA` layout `{x[], y[], z[], vx[], vy[], vz[]}`. Measure NumPy vectorized update speed for both layouts.

6. **Sorting comparison**: Implement bitonic sort and compare it with Python's built-in `sorted()` for arrays of size 256, 1024, and 4096. Discuss when the parallel approach would be faster on a GPU.

## Further Reading

- Kirk, D. and Hwu, W. *Programming Massively Parallel Processors*, 4th ed. Morgan Kaufmann, 2022. (The standard CUDA/GPU computing textbook)
- Harris, M. "Parallel Prefix Sum (Scan) with CUDA." *GPU Gems 3*, NVIDIA, 2007. (Classic GPU scan implementation)
- Akenine-Moller, T. et al. *Real-Time Rendering*, 4th ed. CRC Press, 2018. (Chapter 23: GPU architecture for graphics programmers)
- Sellers, G. *Vulkan Programming Guide*. Addison-Wesley, 2016. (Vulkan compute shaders)
- WebGPU Specification. W3C, 2024. https://www.w3.org/TR/webgpu/ (Web-native GPU computing)
