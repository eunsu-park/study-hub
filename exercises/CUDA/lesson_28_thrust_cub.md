# Lesson 28 — Thrust and CUB (per-lesson exercise)

Prerequisites: L14 (reduction), L15 (scan), L16 (sort).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Thrust is a high-level C++ library providing STL-like algorithms (sort, reduce, transform) for GPU containers. CUB is the lower-level building blocks Thrust uses internally — block- and warp-level primitives you call from your own kernels.

The lesson: when one of these libraries does what you need, USE IT. Hand-rolled implementations rarely beat them and absorb engineering effort better spent elsewhere.

---

## Exercise 28.1 — Thrust Sort

**Difficulty**: ★

### Problem

Sort 16 million `int` values on the GPU using Thrust:

```cuda
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <chrono>
#include <cstdio>

int main(void) {
    int N = 1 << 24;
    thrust::default_random_engine rng(0);
    thrust::uniform_int_distribution<int> dist(0, 1 << 30);

    thrust::host_vector<int> h(N);
    for (int i = 0; i < N; i++) h[i] = dist(rng);

    thrust::device_vector<int> d = h;     /* H2D copy */

    auto t0 = std::chrono::high_resolution_clock::now();
    thrust::sort(d.begin(), d.end());
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("thrust sort %d ints: %.2f ms (%.0f Mkeys/s)\n", N, ms, N / (ms * 1e3));

    /* Verify monotone */
    h = d;
    for (int i = 1; i < N; i++) if (h[i] < h[i-1]) { printf("FAIL\n"); return 1; }
    printf("ok\n");
    return 0;
}
```

For comparison, time `std::sort` on the host. The GPU should be 10–50× faster on $N \geq 10^7$.

---

## Exercise 28.2 — Thrust Reduce / Inclusive Scan

**Difficulty**: ★

Build the GPU equivalents of `std::accumulate` and `std::partial_sum`:

```cuda
int sum  = thrust::reduce(d.begin(), d.end(), 0, thrust::plus<int>());
thrust::inclusive_scan(d.begin(), d.end(), d_out.begin());
```

Compare timings against your hand-rolled CUDA kernels from L14 and L15. Thrust will be within 10% of optimal — sometimes slightly faster, sometimes slightly slower, but never worth re-implementing for production code.

---

## Exercise 28.3 — CUB BlockReduce in a Custom Kernel

**Difficulty**: ★★★

CUB provides `cub::BlockReduce<T, BLOCK_THREADS>` — a templated component you compose into your own kernel:

```cuda
#include <cub/block/block_reduce.cuh>

template <int BLOCK_THREADS>
__global__ void block_sum_kernel(const float *in, float *out, int N) {
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tmp;

    int idx = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    float val = (idx < N) ? in[idx] : 0.0f;
    float block_sum = BlockReduce(tmp).Sum(val);

    if (threadIdx.x == 0) out[blockIdx.x] = block_sum;
}
```

CUB takes care of the warp shuffles and shared-memory shuffling — the boilerplate that makes hand-rolled reductions tedious. Compare against your DL_Scratch_C `reduce_naive` and `reduce_seq` from CUDA L14 — CUB should be at least as fast as your best hand-roll.

---

## Exercise 28.4 — Custom Comparator with Thrust — Bonus

**Difficulty**: ★★

Sort tuples `(int key, float weight)` by `weight` descending. Pass a custom comparator:

```cuda
struct ByWeightDesc {
    __host__ __device__ bool operator()(const thrust::tuple<int, float> &a,
                                        const thrust::tuple<int, float> &b) const {
        return thrust::get<1>(a) > thrust::get<1>(b);
    }
};

thrust::sort(thrust::make_zip_iterator(...), ..., ByWeightDesc());
```

This pattern is the GPU equivalent of `std::sort(v.begin(), v.end(), cmp)`. Used in beam search, ranking, and many other ML inference paths.
