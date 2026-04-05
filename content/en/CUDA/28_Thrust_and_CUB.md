# 28. Thrust and CUB — High-Level GPU Libraries

**Previous**: [Random Number and Stochastic](./27_Random_Number_and_Stochastic.md) | **Next**: [cuBLAS and cuSPARSE](./29_cuBLAS_and_cuSPARSE.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `thrust::device_vector` and Thrust algorithms as GPU STL equivalents
2. Apply Thrust's sort, reduce, scan, transform, and copy_if to practical problems
3. Use `thrust::zip_iterator` to operate on multiple arrays simultaneously
4. Understand CUB's block-level primitives for hand-tuned kernel internals
5. Use CUB device-level APIs (DeviceReduce, DeviceScan, DeviceSort) as drop-in replacements for common patterns

---

## 1. Thrust Basics

Thrust is a C++ template library that provides STL-like algorithms running on the GPU. It is included with CUDA and requires no additional installation.

```cpp
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

// Thrust automatically transfers to/from GPU via assignment
thrust::host_vector<float>   h_vec(1000, 1.0f);  // host, all 1.0
thrust::device_vector<float> d_vec = h_vec;       // H→D copy
thrust::device_vector<float> d_out(1000);

// Raw pointer access (for interop with custom kernels)
float *ptr = thrust::raw_pointer_cast(d_vec.data());
```

---

## 2. Sort

Thrust provides radix sort and merge sort; radix sort is used for arithmetic types:

```cpp
#include <thrust/sort.h>

void thrust_sort_examples() {
    thrust::device_vector<int> keys = {5, 2, 8, 1, 9, 3};

    // Sort in place (ascending)
    thrust::sort(keys.begin(), keys.end());
    // Result: {1, 2, 3, 5, 8, 9}

    // Sort descending
    thrust::sort(keys.begin(), keys.end(), thrust::greater<int>());

    // Sort by key with paired values (key-value sort)
    thrust::device_vector<int>   k = {3, 1, 4, 1, 5, 9};
    thrust::device_vector<float> v = {3.f, 1.f, 4.f, 1.f, 5.f, 9.f};
    thrust::sort_by_key(k.begin(), k.end(), v.begin());
    // k sorted; v rearranged to match

    // Stable sort (preserves relative order of equal elements)
    thrust::stable_sort(keys.begin(), keys.end());
}
```

---

## 3. Reduce

```cpp
#include <thrust/reduce.h>

void thrust_reduce_examples() {
    thrust::device_vector<float> v(1000000);
    thrust::fill(v.begin(), v.end(), 1.0f);

    // Sum (default binary op = thrust::plus<float>)
    float total = thrust::reduce(v.begin(), v.end(), 0.0f);
    // total = 1,000,000.0

    // Min / max
    float vmin = thrust::reduce(v.begin(), v.end(),  1e30f, thrust::minimum<float>());
    float vmax = thrust::reduce(v.begin(), v.end(), -1e30f, thrust::maximum<float>());

    // reduce_by_key: segmented reduce
    thrust::device_vector<int>   keys   = {0, 0, 1, 1, 2};
    thrust::device_vector<float> vals   = {1, 2, 3, 4, 5};
    thrust::device_vector<int>   out_k(5);
    thrust::device_vector<float> out_v(5);
    auto end = thrust::reduce_by_key(keys.begin(), keys.end(),
                                      vals.begin(),
                                      out_k.begin(), out_v.begin());
    // out_k: {0, 1, 2}, out_v: {3, 7, 5}
    int n_segments = end.first - out_k.begin();
    printf("%d segments\n", n_segments);
}
```

---

## 4. Transform

`thrust::transform` applies a functor element-wise — equivalent to `std::transform` but on the GPU:

```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>

// Custom functor: scale + offset
struct ScaleOffset {
    float scale, offset;
    __host__ __device__
    float operator()(float x) const { return scale * x + offset; }
};

void thrust_transform_examples() {
    thrust::device_vector<float> a(1000, 2.f);
    thrust::device_vector<float> b(1000, 3.f);
    thrust::device_vector<float> c(1000);

    // Unary transform: c[i] = a[i] * 2 + 1
    thrust::transform(a.begin(), a.end(), c.begin(), ScaleOffset{2.f, 1.f});

    // Binary transform: c[i] = a[i] + b[i]
    thrust::transform(a.begin(), a.end(), b.begin(), c.begin(),
                      thrust::plus<float>());

    // Fused transform + reduce (inner dot product)
    float dot = thrust::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    // Equivalent to Σ a[i]*b[i]
}
```

---

## 5. Scan (Prefix Sum)

```cpp
#include <thrust/scan.h>

void thrust_scan_examples() {
    thrust::device_vector<int> v = {1, 2, 3, 4, 5};
    thrust::device_vector<int> out(5);

    // Exclusive scan: out[i] = Σ_{j<i} v[j]
    thrust::exclusive_scan(v.begin(), v.end(), out.begin(), 0);
    // out: {0, 1, 3, 6, 10}

    // Inclusive scan: out[i] = Σ_{j<=i} v[j]
    thrust::inclusive_scan(v.begin(), v.end(), out.begin());
    // out: {1, 3, 6, 10, 15}

    // Segmented scan (scan resets at each key change)
    thrust::device_vector<int> keys = {0, 0, 1, 1, 2};
    thrust::inclusive_scan_by_key(keys.begin(), keys.end(),
                                   v.begin(), out.begin());
    // out: {1, 3, 3, 7, 5}
}
```

---

## 6. copy_if (Stream Compaction)

```cpp
#include <thrust/copy.h>

// Select elements satisfying a predicate (stream compaction)
struct IsPositive {
    __host__ __device__
    bool operator()(float x) const { return x > 0.f; }
};

void thrust_copy_if_example() {
    thrust::device_vector<float> src = {-1, 2, -3, 4, -5, 6};
    thrust::device_vector<float> dst(src.size());

    auto end = thrust::copy_if(src.begin(), src.end(), dst.begin(), IsPositive{});
    dst.resize(end - dst.begin());
    // dst: {2, 4, 6}

    // Count matching elements
    int n = thrust::count_if(src.begin(), src.end(), IsPositive{});
    printf("%d positive elements\n", n);  // 3
}
```

---

## 7. zip_iterator — Multiple Array Operations

`zip_iterator` treats multiple arrays as a single range of tuples, enabling multi-field sort or transform:

```cpp
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

void zip_iterator_example() {
    // Sort an array of (x, y) coordinates by distance from origin
    thrust::device_vector<float> x = {3, 1, 4, 1, 5};
    thrust::device_vector<float> y = {4, 0, 3, 1, 12};

    // Custom comparator: sort by x²+y²
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin()));
    auto zip_end   = thrust::make_zip_iterator(thrust::make_tuple(x.end(),   y.end()));

    thrust::sort(zip_begin, zip_end, [] __host__ __device__
        (const thrust::tuple<float,float>& a,
         const thrust::tuple<float,float>& b) {
        float da = thrust::get<0>(a)*thrust::get<0>(a) + thrust::get<1>(a)*thrust::get<1>(a);
        float db = thrust::get<0>(b)*thrust::get<0>(b) + thrust::get<1>(b)*thrust::get<1>(b);
        return da < db;
    });
    // x and y sorted together by distance from origin
}
```

---

## 8. CUB Block-Level Primitives

CUB (CUDA UnBound) provides warp/block/device level primitives for use inside custom kernels. Block-level operations use shared memory automatically:

```cpp
#include <cub/cub.cuh>

// Reduction within a thread block
__global__ void block_reduce_demo(const float *in, float *out, int N) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp;

    int i = blockIdx.x * 256 + threadIdx.x;
    float val = (i < N) ? in[i] : 0.f;

    // Sum across all threads in block → result only valid in thread 0
    float block_sum = BlockReduce(temp).Sum(val);

    if (threadIdx.x == 0) out[blockIdx.x] = block_sum;
}

// Scan within a thread block
__global__ void block_scan_demo(const int *in, int *out, int N) {
    using BlockScan = cub::BlockScan<int, 128>;
    __shared__ typename BlockScan::TempStorage temp;

    int i = blockIdx.x * 128 + threadIdx.x;
    int val = (i < N) ? in[i] : 0;

    int prefix_sum;
    BlockScan(temp).ExclusiveSum(val, prefix_sum);  // per-thread output

    if (i < N) out[i] = prefix_sum;
}

// Warp-level: cub::WarpReduce (no shared memory required)
__global__ void warp_reduce_demo(const float *in, float *out, int N) {
    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp[4];  // 4 warps per block

    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = threadIdx.x / 32;
    float val = (i < N) ? in[i] : 0.f;

    float wsum = WarpReduce(temp[warp]).Sum(val);

    if (threadIdx.x % 32 == 0) out[blockIdx.x * 4 + warp] = wsum;
}
```

---

## 9. CUB Device-Level APIs

CUB device-level functions handle the full problem (all blocks, entire array) and manage temporary storage internally:

```cpp
// DeviceReduce: single-call reduction
void cub_device_reduce(const float *d_in, float *d_out, int N) {
    void   *d_temp = nullptr;
    size_t  temp_bytes = 0;

    // Step 1: query temp storage size
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);

    // Step 2: allocate temp storage
    cudaMalloc(&d_temp, temp_bytes);

    // Step 3: run
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);
    cudaFree(d_temp);
}

// DeviceScan: exclusive prefix sum
void cub_device_scan(const int *d_in, int *d_out, int N) {
    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, N);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, N);
    cudaFree(d_temp);
}

// DeviceRadixSort: fastest GPU sort for arithmetic types
void cub_sort(int *d_keys, int *d_vals, int N) {
    cub::DoubleBuffer<int> d_keys_buf(d_keys, nullptr);
    cub::DoubleBuffer<int> d_vals_buf(d_vals, nullptr);
    // Need to allocate alternate buffers...
    // Simpler form:
    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                                     d_keys, d_keys,   // in-place not directly supported
                                     d_vals, d_vals, N);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                                     d_keys, d_keys,
                                     d_vals, d_vals, N);
    cudaFree(d_temp);
}
```

---

## 10. When to Use Thrust vs CUB vs Custom Kernels

```
Thrust:
  + Easy to use, STL-like interface
  + Good for algorithm prototyping
  + Works on host and device vectors
  - Less flexible for complex custom logic
  - Higher launch overhead for small N

CUB Block/Warp primitives (inside custom kernels):
  + Direct control over shared memory and registers
  + Optimal for complex kernels needing embedded reductions/scans
  + Zero overhead: primitives inline directly into kernel

CUB Device-level:
  + Production-quality implementations (near-optimal FLOPS/bandwidth)
  + Two-call pattern (query size, then run)
  + Prefer over Thrust for performance-critical standalone operations

Custom CUDA kernels:
  + Required when no library primitive fits
  + Use CUB primitives internally where possible
  - Most development time; profile before optimizing

Rule of thumb:
  Prototype with Thrust → profile → replace bottlenecks with CUB/custom
```

---

## Key Takeaways

- **Thrust** provides GPU-accelerated STL equivalents (sort, reduce, scan, transform, copy_if) with `device_vector` managing memory automatically
- **thrust::sort_by_key** sorts key-value pairs; **thrust::reduce_by_key** performs segmented reductions on consecutive equal keys
- **thrust::zip_iterator** treats multiple arrays as a single range of tuples, enabling multi-field operations without AoS conversion
- **CUB BlockReduce / BlockScan** are embedded in custom kernels: allocate `TempStorage` in shared memory, construct the object, call `.Sum()` or `.ExclusiveSum()`
- **CUB DeviceReduce / DeviceScan / DeviceRadixSort** are standalone routines: call twice (first to query temp size, second to run)
- CUB's DeviceRadixSort is the fastest GPU sort for integer and float types, typically outperforming thrust::sort by 10-30% for large N

---

**Next**: [29. cuBLAS and cuSPARSE](./29_cuBLAS_and_cuSPARSE.md) — Accelerate dense and sparse matrix operations using NVIDIA's cuBLAS (BLAS) and cuSPARSE libraries, including Tensor Core GEMM and CSR sparse matrix-vector multiplication.
