# 28. Thrust와 CUB — 고수준 GPU 라이브러리

**이전**: [Random Number and Stochastic](./27_Random_Number_and_Stochastic.md) | **다음**: [cuBLAS and cuSPARSE](./29_cuBLAS_and_cuSPARSE.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. GPU STL 동등물로서 `thrust::device_vector`와 Thrust 알고리즘 사용하기
2. 실용적인 문제에 Thrust의 sort, reduce, scan, transform, copy_if 적용하기
3. 여러 배열을 동시에 조작하기 위해 `thrust::zip_iterator` 사용하기
4. 직접 튜닝된 kernel 내부를 위한 CUB의 block 수준 기본 요소 이해하기
5. CUB device 수준 API (DeviceReduce, DeviceScan, DeviceSort)를 일반적인 패턴의 드롭인 대체제로 사용하기

---

## 1. Thrust 기초

Thrust는 GPU에서 실행되는 STL 유사 알고리즘을 제공하는 C++ 템플릿 라이브러리입니다. CUDA에 포함되어 있으며 추가 설치가 필요 없습니다.

```cpp
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

// Thrust는 할당을 통해 자동으로 GPU로/에서 전송
thrust::host_vector<float>   h_vec(1000, 1.0f);  // 호스트, 모두 1.0
thrust::device_vector<float> d_vec = h_vec;       // H→D 복사
thrust::device_vector<float> d_out(1000);

// 원시 포인터 접근 (커스텀 kernel과의 상호 운용)
float *ptr = thrust::raw_pointer_cast(d_vec.data());
```

---

## 2. Sort (정렬)

Thrust는 radix sort와 merge sort를 제공합니다; 산술 타입에는 radix sort가 사용됩니다:

```cpp
#include <thrust/sort.h>

void thrust_sort_examples() {
    thrust::device_vector<int> keys = {5, 2, 8, 1, 9, 3};

    // 제자리 정렬 (오름차순)
    thrust::sort(keys.begin(), keys.end());
    // 결과: {1, 2, 3, 5, 8, 9}

    // 내림차순 정렬
    thrust::sort(keys.begin(), keys.end(), thrust::greater<int>());

    // 쌍 값으로 키 정렬 (key-value 정렬)
    thrust::device_vector<int>   k = {3, 1, 4, 1, 5, 9};
    thrust::device_vector<float> v = {3.f, 1.f, 4.f, 1.f, 5.f, 9.f};
    thrust::sort_by_key(k.begin(), k.end(), v.begin());
    // k 정렬됨; v가 일치하도록 재배열됨

    // 안정 정렬 (동일 요소의 상대적 순서 보존)
    thrust::stable_sort(keys.begin(), keys.end());
}
```

---

## 3. Reduce (축소)

```cpp
#include <thrust/reduce.h>

void thrust_reduce_examples() {
    thrust::device_vector<float> v(1000000);
    thrust::fill(v.begin(), v.end(), 1.0f);

    // 합 (기본 이진 연산 = thrust::plus<float>)
    float total = thrust::reduce(v.begin(), v.end(), 0.0f);
    // total = 1,000,000.0

    // 최솟값 / 최댓값
    float vmin = thrust::reduce(v.begin(), v.end(),  1e30f, thrust::minimum<float>());
    float vmax = thrust::reduce(v.begin(), v.end(), -1e30f, thrust::maximum<float>());

    // reduce_by_key: 세그먼트 reduce
    thrust::device_vector<int>   keys   = {0, 0, 1, 1, 2};
    thrust::device_vector<float> vals   = {1, 2, 3, 4, 5};
    thrust::device_vector<int>   out_k(5);
    thrust::device_vector<float> out_v(5);
    auto end = thrust::reduce_by_key(keys.begin(), keys.end(),
                                      vals.begin(),
                                      out_k.begin(), out_v.begin());
    // out_k: {0, 1, 2}, out_v: {3, 7, 5}
    int n_segments = end.first - out_k.begin();
    printf("%d 세그먼트\n", n_segments);
}
```

---

## 4. Transform (변환)

`thrust::transform`은 GPU에서 functor를 요소별로 적용합니다 — `std::transform`과 동등하지만 GPU에서 실행됩니다:

```cpp
#include <thrust/transform.h>
#include <thrust/functional.h>

// 커스텀 functor: 스케일 + 오프셋
struct ScaleOffset {
    float scale, offset;
    __host__ __device__
    float operator()(float x) const { return scale * x + offset; }
};

void thrust_transform_examples() {
    thrust::device_vector<float> a(1000, 2.f);
    thrust::device_vector<float> b(1000, 3.f);
    thrust::device_vector<float> c(1000);

    // 단항 변환: c[i] = a[i] * 2 + 1
    thrust::transform(a.begin(), a.end(), c.begin(), ScaleOffset{2.f, 1.f});

    // 이항 변환: c[i] = a[i] + b[i]
    thrust::transform(a.begin(), a.end(), b.begin(), c.begin(),
                      thrust::plus<float>());

    // 융합 변환 + reduce (내적)
    float dot = thrust::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    // Σ a[i]*b[i]와 동등
}
```

---

## 5. Scan (prefix sum)

```cpp
#include <thrust/scan.h>

void thrust_scan_examples() {
    thrust::device_vector<int> v = {1, 2, 3, 4, 5};
    thrust::device_vector<int> out(5);

    // 배타적 scan: out[i] = Σ_{j<i} v[j]
    thrust::exclusive_scan(v.begin(), v.end(), out.begin(), 0);
    // out: {0, 1, 3, 6, 10}

    // 포괄적 scan: out[i] = Σ_{j<=i} v[j]
    thrust::inclusive_scan(v.begin(), v.end(), out.begin());
    // out: {1, 3, 6, 10, 15}

    // 세그먼트 scan (각 키 변경 시 scan 재시작)
    thrust::device_vector<int> keys = {0, 0, 1, 1, 2};
    thrust::inclusive_scan_by_key(keys.begin(), keys.end(),
                                   v.begin(), out.begin());
    // out: {1, 3, 3, 7, 5}
}
```

---

## 6. copy_if (스트림 압축)

```cpp
#include <thrust/copy.h>

// 술어를 만족하는 요소 선택 (스트림 압축)
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

    // 일치하는 요소 집계
    int n = thrust::count_if(src.begin(), src.end(), IsPositive{});
    printf("%d 양수 요소\n", n);  // 3
}
```

---

## 7. zip_iterator — 다중 배열 연산

`zip_iterator`는 여러 배열을 단일 튜플 범위로 처리하여 다중 필드 정렬이나 변환을 가능하게 합니다:

```cpp
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

void zip_iterator_example() {
    // (x, y) 좌표 배열을 원점으로부터의 거리로 정렬
    thrust::device_vector<float> x = {3, 1, 4, 1, 5};
    thrust::device_vector<float> y = {4, 0, 3, 1, 12};

    // 커스텀 비교기: x²+y²으로 정렬
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin()));
    auto zip_end   = thrust::make_zip_iterator(thrust::make_tuple(x.end(),   y.end()));

    thrust::sort(zip_begin, zip_end, [] __host__ __device__
        (const thrust::tuple<float,float>& a,
         const thrust::tuple<float,float>& b) {
        float da = thrust::get<0>(a)*thrust::get<0>(a) + thrust::get<1>(a)*thrust::get<1>(a);
        float db = thrust::get<0>(b)*thrust::get<0>(b) + thrust::get<1>(b)*thrust::get<1>(b);
        return da < db;
    });
    // x와 y가 원점으로부터의 거리로 함께 정렬됨
}
```

---

## 8. CUB Block 수준 기본 요소

CUB (CUDA UnBound)는 커스텀 kernel 내부에서 사용하기 위한 warp/block/device 수준 기본 요소를 제공합니다. Block 수준 연산은 shared memory를 자동으로 사용합니다:

```cpp
#include <cub/cub.cuh>

// thread block 내의 reduction
__global__ void block_reduce_demo(const float *in, float *out, int N) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp;

    int i = blockIdx.x * 256 + threadIdx.x;
    float val = (i < N) ? in[i] : 0.f;

    // block의 모든 thread에 대한 합 → thread 0에서만 유효한 결과
    float block_sum = BlockReduce(temp).Sum(val);

    if (threadIdx.x == 0) out[blockIdx.x] = block_sum;
}

// thread block 내의 scan
__global__ void block_scan_demo(const int *in, int *out, int N) {
    using BlockScan = cub::BlockScan<int, 128>;
    __shared__ typename BlockScan::TempStorage temp;

    int i = blockIdx.x * 128 + threadIdx.x;
    int val = (i < N) ? in[i] : 0;

    int prefix_sum;
    BlockScan(temp).ExclusiveSum(val, prefix_sum);  // thread당 출력

    if (i < N) out[i] = prefix_sum;
}

// Warp 수준: cub::WarpReduce (shared memory 불필요)
__global__ void warp_reduce_demo(const float *in, float *out, int N) {
    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp[4];  // block당 4개 warp

    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = threadIdx.x / 32;
    float val = (i < N) ? in[i] : 0.f;

    float wsum = WarpReduce(temp[warp]).Sum(val);

    if (threadIdx.x % 32 == 0) out[blockIdx.x * 4 + warp] = wsum;
}
```

---

## 9. CUB Device 수준 API

CUB device 수준 함수는 전체 문제 (모든 block, 전체 배열)를 처리하고 임시 저장소를 내부적으로 관리합니다:

```cpp
// DeviceReduce: 단일 호출 reduction
void cub_device_reduce(const float *d_in, float *d_out, int N) {
    void   *d_temp = nullptr;
    size_t  temp_bytes = 0;

    // 단계 1: 임시 저장소 크기 조회
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);

    // 단계 2: 임시 저장소 할당
    cudaMalloc(&d_temp, temp_bytes);

    // 단계 3: 실행
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, N);
    cudaFree(d_temp);
}

// DeviceScan: 배타적 prefix sum
void cub_device_scan(const int *d_in, int *d_out, int N) {
    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, N);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, N);
    cudaFree(d_temp);
}

// DeviceRadixSort: 산술 타입에 대한 가장 빠른 GPU 정렬
void cub_sort(int *d_keys, int *d_vals, int N) {
    cub::DoubleBuffer<int> d_keys_buf(d_keys, nullptr);
    cub::DoubleBuffer<int> d_vals_buf(d_vals, nullptr);
    // 대체 버퍼를 할당해야 함...
    // 더 단순한 형태:
    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                                     d_keys, d_keys,   // in-place는 직접 지원 안 됨
                                     d_vals, d_vals, N);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                                     d_keys, d_keys,
                                     d_vals, d_vals, N);
    cudaFree(d_temp);
}
```

---

## 10. Thrust vs CUB vs 커스텀 Kernel 선택 기준

```
Thrust:
  + 사용하기 쉬움, STL 유사 인터페이스
  + 알고리즘 프로토타이핑에 적합
  + 호스트 및 device 벡터에서 동작
  - 복잡한 커스텀 로직에 유연성 낮음
  - 작은 N에서 실행 오버헤드 높음

CUB Block/Warp 기본 요소 (커스텀 kernel 내부):
  + shared memory와 register에 대한 직접 제어
  + 내장 reduction/scan이 필요한 복잡한 kernel에 최적
  + 오버헤드 없음: 기본 요소가 kernel에 직접 인라인됨

CUB Device 수준:
  + 프로덕션 품질 구현 (최적에 근접한 FLOPS/대역폭)
  + 2-호출 패턴 (크기 조회, 그 다음 실행)
  + 성능 중요 독립 연산에서 Thrust보다 선호

커스텀 CUDA kernel:
  + 라이브러리 기본 요소가 맞지 않을 때 필요
  + 가능하면 내부적으로 CUB 기본 요소 사용
  - 개발 시간이 가장 많이 걸림; 최적화 전에 프로파일링

경험 법칙:
  Thrust로 프로토타입 → 프로파일링 → CUB/커스텀으로 병목 교체
```

---

## 핵심 요약

- **Thrust**는 GPU 가속 STL 동등물(sort, reduce, scan, transform, copy_if)을 제공하며 `device_vector`가 메모리를 자동 관리합니다
- **thrust::sort_by_key**는 키-값 쌍을 정렬합니다; **thrust::reduce_by_key**는 연속된 동일 키에 대해 세그먼트 reduction을 수행합니다
- **thrust::zip_iterator**는 여러 배열을 단일 튜플 범위로 처리하여 AoS 변환 없이 다중 필드 연산을 가능하게 합니다
- **CUB BlockReduce / BlockScan**은 커스텀 kernel에 내장됩니다: shared memory에 `TempStorage`를 할당하고, 객체를 생성하고, `.Sum()` 또는 `.ExclusiveSum()` 호출
- **CUB DeviceReduce / DeviceScan / DeviceRadixSort**는 독립 루틴입니다: 두 번 호출 (먼저 임시 크기 조회, 그 다음 실행)
- CUB의 DeviceRadixSort는 정수와 float 타입에 대한 가장 빠른 GPU 정렬로, 대형 N에서 thrust::sort보다 일반적으로 10-30% 더 빠릅니다

---

**다음**: [29. cuBLAS and cuSPARSE](./29_cuBLAS_and_cuSPARSE.md) — Tensor Core GEMM과 CSR 희소 행렬-벡터 곱셈을 포함하여 NVIDIA의 cuBLAS (BLAS)와 cuSPARSE 라이브러리를 사용하여 밀집 및 희소 행렬 연산을 가속합니다.
