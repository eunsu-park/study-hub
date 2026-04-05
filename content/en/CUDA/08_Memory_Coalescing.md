# 08. Memory Coalescing

**Previous**: [Atomic Operations](./07_Atomic_Operations.md) | **Next**: [Occupancy and Launch Configuration](./09_Occupancy_and_Launch_Config.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the 128-byte transaction granularity rule and its implications
2. Identify coalesced vs non-coalesced access patterns visually
3. Measure the stride penalty using a benchmark kernel
4. Restructure Array-of-Structures (AoS) to Structure-of-Arrays (SoA) for coalescing
5. Use Nsight Compute to confirm coalescing efficiency

---

## 1. The 128-Byte Transaction Rule

When a warp of 32 threads accesses global memory, the hardware combines requests into **128-byte transactions** (one cache line). The number of transactions required depends entirely on the access pattern:

```
32 threads × 4 bytes (float) = 128 bytes total

Best case (coalesced): all 32 threads access consecutive addresses
  → 1 transaction for the entire warp  ✓

Worst case (stride = 1 element apart, fully coalesced):
  Thread 0: addr 0
  Thread 1: addr 4
  Thread 2: addr 8
  ...
  Thread 31: addr 124
  → All fit in one 128-byte cache line → 1 transaction ✓

Stride-2 access:
  Thread 0: addr 0
  Thread 1: addr 8
  Thread 2: addr 16
  ...
  Thread 31: addr 248
  → Spans 2 cache lines → 2 transactions (50% efficiency)

Stride-32 (worst):
  Thread 0: addr 0
  Thread 1: addr 128
  Thread 2: addr 256
  ...
  → Every thread in a different cache line → 32 transactions (3% efficiency)
```

---

## 2. Visual: Coalesced vs Strided Access

```
Memory layout: [ 0 ][ 1 ][ 2 ][ 3 ]...[ 31 ][ 32 ]...[ 63 ]
                ←────────── cache line 0 ──────────→

Coalesced: thread k accesses element k
  T0→[0], T1→[1], ..., T31→[31]
  ──────────────────────────────────
  1 cache line loaded → 1 transaction → 100% efficiency

Strided (stride=2): thread k accesses element 2k
  T0→[0], T1→[2], T2→[4], ..., T15→[30] | T16→[32], T17→[34], ..., T31→[62]
  ──────────────────────────────────────────────────────────────────────────
  Occupies 2 cache lines → 2 transactions, but only uses 16 bytes from each
  50% efficiency (2× bandwidth wasted)

Strided (stride=32): thread k accesses element 32k
  T0→[0], T1→[32], ..., T31→[992]
  Each element in a different cache line → 32 transactions
  3.1% efficiency (32× bandwidth wasted)
```

---

## 3. Stride Benchmark

```c
// benchmark_stride.cu
__global__ void stride_read(const float *data, float *result, int stride, long n) {
    long i = (long)(blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (i < n) {
        result[blockIdx.x * blockDim.x + threadIdx.x] = data[i];
    }
}

// For each stride value, measure effective bandwidth:
// stride=1 (coalesced):  ~2000 GB/s  (A100 peak)
// stride=2:              ~1000 GB/s  (2 transactions per warp)
// stride=4:              ~ 500 GB/s
// stride=8:              ~ 250 GB/s
// stride=32:             ~  62 GB/s  (32 transactions per warp, ~3% of peak)
```

The bandwidth degrades **linearly with stride** — each doubling of stride halves effective bandwidth.

---

## 4. Array-of-Structures (AoS) vs Structure-of-Arrays (SoA)

This is the most common coalescing design decision in GPU code.

### AoS Layout (bad for GPU)

```c
struct Particle {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
    float mass;
};

Particle particles[N];  // AoS: [x0,y0,z0,vx0,vy0,vz0,m0, x1,y1,z1,...]

// Kernel accessing x:
float px = particles[tid].x;
// Thread 0: address = 0   (x0)
// Thread 1: address = 28  (x1, stride = sizeof(Particle) = 28 bytes)
// Thread 2: address = 56  (x2)
// → Stride = 7 floats → 7× wasted bandwidth
```

### SoA Layout (good for GPU)

```c
struct ParticlesSoA {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *mass;
};

ParticlesSoA p;  // SoA: [x0,x1,x2,...,xN, y0,y1,...,yN, ...]

// Kernel accessing x:
float px = p.x[tid];
// Thread 0: address = 0  (x0)
// Thread 1: address = 4  (x1)
// Thread 2: address = 8  (x2)
// → Stride = 1 float → 1 transaction → 100% efficiency ✓
```

### Complete Example: N-Body Force Calculation (AoS → SoA)

```c
// AoS version (baseline)
struct Body { float x, y, z, mass; };

__global__ void force_aos(const Body *bodies, float3 *forces, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 f = {0.0f, 0.0f, 0.0f};
    float xi = bodies[i].x;  // stride = sizeof(Body) = 16 bytes
    float yi = bodies[i].y;
    float zi = bodies[i].z;
    for (int j = 0; j < N; j++) {
        float dx = bodies[j].x - xi;  // stride access for each j
        float dy = bodies[j].y - yi;
        float dz = bodies[j].z - zi;
        float r  = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f);
        float f_mag = bodies[j].mass / (r * r * r);
        f.x += dx * f_mag;
        f.y += dy * f_mag;
        f.z += dz * f_mag;
    }
    forces[i] = f;
}

// SoA version (optimized)
__global__ void force_soa(
    const float *x, const float *y, const float *z, const float *mass,
    float *fx, float *fy, float *fz, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float xi = x[i], yi = y[i], zi = z[i];  // coalesced reads ✓
    float fix = 0, fiy = 0, fiz = 0;
    for (int j = 0; j < N; j++) {
        float dx = x[j] - xi;  // sequential reads in j loop (L1 cached)
        float dy = y[j] - yi;
        float dz = z[j] - zi;
        float r  = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f);
        float f_mag = mass[j] / (r * r * r);
        fix += dx * f_mag;
        fiy += dy * f_mag;
        fiz += dz * f_mag;
    }
    fx[i] = fix; fy[i] = fiy; fz[i] = fiz;  // coalesced writes ✓
}
```

Typical speedup: **2–4×** for this type of kernel.

---

## 5. Matrix Access: Row vs Column

When accessing a 2D matrix:

```c
// Row access (coalesced for row-major): thread k reads row 0, column k
float val = matrix[0 * N + tid];   // consecutive → coalesced ✓

// Column access (strided for row-major): thread k reads row k, column 0
float val = matrix[tid * N + 0];   // stride N → highly non-coalesced ✗
```

For column access, the fix is to **transpose before processing** or use the **tiled transpose** technique from L05.

---

## 6. Vectorized Loads: `float4`

Load 4 floats in a single instruction — increases instruction efficiency:

```c
// Scalar load (4 memory instructions for 4 floats)
float a0 = data[4*i + 0];
float a1 = data[4*i + 1];
float a2 = data[4*i + 2];
float a3 = data[4*i + 3];

// Vectorized load (1 memory instruction, 128-bit)
float4 v = reinterpret_cast<float4*>(data)[i];
float a0 = v.x, a1 = v.y, a2 = v.z, a3 = v.w;
```

Requirements:
- Data must be 16-byte aligned (floats: `cudaMalloc` guarantees 256-byte alignment)
- Total elements must be multiple of 4

This typically gives **5–15% speedup** on memory-bound kernels by reducing instruction overhead.

---

## 7. Profiling Coalescing with Nsight Compute

```bash
ncu --metrics \
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    ./my_kernel

# Key metrics:
# sectors_per_request = 1.0 → perfect coalescing (1 cache line per warp)
# sectors_per_request = 32  → fully uncoalesced (32 cache lines per warp)
# sectors_per_request = 4   → 4× wasted bandwidth
```

The `l1tex__average_t_sectors_per_request` ratio is the most direct measure: **1.0 is optimal, 32 is worst case**.

---

## 8. Shared Memory as a Coalescing Shim

When the access pattern can't be made coalesced in global memory (e.g., a matrix accessed by column), use shared memory as an intermediate step:

```c
// Load coalesced into shared memory, then read in any pattern
__shared__ float smem[256];

// Coalesced global → shared load
smem[threadIdx.x] = global_data[coalesced_index];
__syncthreads();

// Non-coalesced pattern in shared memory (fast — no DRAM)
float val = smem[shuffled_index(threadIdx.x)];
```

This is exactly the tiled transpose from L05 — the key insight is: **global memory must be coalesced; shared memory access patterns don't matter (except for bank conflicts)**.

---

## Key Takeaways

- The GPU issues memory in **128-byte (32-element) transactions** — one warp, one cache line is the ideal
- **Stride-1** (consecutive) access = 1 transaction per warp = full bandwidth
- **Stride-N** access = N transactions per warp = 1/N bandwidth
- **SoA is better than AoS** for GPU — transforms struct field access from stride-N to stride-1
- Use `float4`/`float2` loads to reduce instruction count on memory-bound kernels
- Profile with `l1tex__average_t_sectors_per_request` — target 1.0

---

**Next**: [09. Occupancy and Launch Configuration](./09_Occupancy_and_Launch_Config.md) — Quantify how register pressure and shared memory limits constrain occupancy, and use `__launch_bounds__` to guide the compiler.
