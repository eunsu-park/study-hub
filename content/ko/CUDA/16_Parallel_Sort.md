# 16. Parallel Sort

**이전**: [Parallel Scan Prefix Sum](./15_Parallel_Scan_Prefix_Sum.md) | **다음**: [Stencil Computations](./17_Stencil_Computations.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 2의 거듭제곱 입력 크기에 대해 GPU에서 bitonic sort 구현하기
2. Radix sort가 scan을 사용해 O(n) 정렬 pass를 달성하는 방법 설명하기
3. 프로덕션 정렬을 위해 CUB `DeviceRadixSort`와 `thrust::sort` 사용하기
4. 데이터 타입, 크기, 접근 패턴에 따라 적절한 정렬 알고리즘 선택하기
5. 초당 요소 수로 정렬 처리량 측정하고 이론적 한계와 비교하기

---

## 1. GPU에서 정렬이 어려운 이유

정렬은 본질적으로 **데이터 의존적**입니다 — 메모리 접근과 비교 순서가 데이터 값에 따라 달라집니다. 이는 warp의 모든 thread가 동일한 명령을 실행하는 GPU의 SIMD 실행 모델과 충돌합니다.

좋은 GPU 정렬 알고리즘은 두 가지 특성을 활용합니다:
1. **Oblivious 비교기 네트워크** (bitonic, odd-even merge): 비교 순서가 데이터 값에 관계없이 고정 — divergence 없음
2. **자릿수 분해** (radix sort): 정렬을 카운팅 + prefix sum pass 시퀀스로 축소, 각 pass는 자명하게 병렬화 가능

---

## 2. Bitonic Sort

Bitonic sort는 **정렬 네트워크**입니다 — 모든 입력을 올바르게 정렬하는 고정된 비교-교환 연산 시퀀스입니다. 시퀀스가 데이터 독립적이므로 모든 thread가 동일한 명령 경로를 따릅니다 (divergence 없음).

**Bitonic 시퀀스**: 처음에 증가한 다음 감소하는 시퀀스 (또는 그 반대).

```
N=8 예시: 각 단계에서의 비교-교환 쌍
Pass 1 (k=2): [0↔1, 2↔3, 4↔5, 6↔7]  (4개의 bitonic 쌍 형성)
Pass 2 (k=4): [0↔3, 1↔2] 그 다음 [4↔7, 5↔6]
Pass 3 (k=8): [0↔7, 1↔6, 2↔5, 3↔4] (정렬된 시퀀스로 병합)
```

```c
// Bitonic sort — N개 요소 배열 정렬 (N은 2의 거듭제곱이어야 함)
__device__ void compare_and_swap(float *a, float *b, bool ascending) {
    if (ascending ? (*a > *b) : (*a < *b)) {
        float tmp = *a; *a = *b; *b = tmp;
    }
}

__global__ void bitonic_sort_step(float *data, int j, int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ij = i ^ j;

    if (ij > i) {
        // 어떤 bitonic 시퀀스를 병합하는지에 따라 정렬 방향 결정
        bool ascending = ((i & k) == 0);
        compare_and_swap(&data[i], &data[ij], ascending);
    }
}

// 호스트: log2(N)*(log2(N)+1)/2번의 kernel pass 실행
void bitonic_sort(float *d_data, int n) {
    // n은 2의 거듭제곱이어야 함
    const int BLOCK = 256;
    int grid = n / BLOCK;

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<grid, BLOCK>>>(d_data, j, k);
            cudaDeviceSynchronize();
        }
    }
}
```

**소규모 부배열(block에 맞는)을 위한 shared memory 최적화**:

```c
// shared memory를 사용해 block 내에서 정렬 — stride < blockDim.x인 내부 pass에서
// global memory 왕복 방지
__global__ void bitonic_sort_shared(float *data, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < n) ? data[gid] : FLT_MAX;
    __syncthreads();

    int bsize = blockDim.x;
    for (int k = 2; k <= bsize; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ij = tid ^ j;
            if (ij > tid) {
                bool asc = ((tid & k) == 0);
                if (asc ? (sdata[tid] > sdata[ij]) : (sdata[tid] < sdata[ij])) {
                    float tmp = sdata[tid]; sdata[tid] = sdata[ij]; sdata[ij] = tmp;
                }
            }
            __syncthreads();
        }
    }

    if (gid < n) data[gid] = sdata[tid];
}
```

**복잡도**: O(n log²n) 비교, O(log²n) 병렬 단계. GPU에서 n ≤ 100만에 실용적.

---

## 3. Radix Sort

Radix sort는 한 번에 b비트를 처리합니다 (일반적으로 4비트 pass = 16개 bucket). 각 pass:
1. 각 bucket에 몇 개의 요소가 있는지 **카운팅**
2. bucket 오프셋을 구하기 위해 카운트의 **배타적 scan**
3. 새로운 위치로 요소 **scatter**

총 작업량은 O(n * 32/b) — 32비트 정수에 4비트 pass: 8 pass.

```c
// 1비트 radix sort pass (명확성을 위해; 프로덕션은 4비트 사용)
__global__ void radix_1bit_pass(const uint32_t *in, uint32_t *out,
                                int *zeros_count, int n, int bit) {
    // Phase 1: 'bit' 위치에 0 또는 1이 있는 요소 판별
    extern __shared__ int sdata[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < n) ? ((in[tid] >> bit) & 1) : 1;

    sdata[threadIdx.x] = (val == 0) ? 1 : 0;  // 0-bucket 플래그
    __syncthreads();

    // 지역 위치를 구하기 위해 block 내 Blelloch scan
    // (간소화 — 프로덕션 CUB가 올바르게 처리)
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int x = (threadIdx.x >= stride) ? sdata[threadIdx.x - stride] : 0;
        __syncthreads();
        sdata[threadIdx.x] += x;
        __syncthreads();
    }

    // Block 수준 scatter (실제 멀티-block 조정은 간결성을 위해 생략)
    if (tid < n) {
        // 위치는 block scan + inter-block prefix에서 계산
        // out[position] = in[tid];
    }

    // 이 block의 0 수 보고 (inter-block scan을 위해)
    if (threadIdx.x == blockDim.x - 1)
        atomicAdd(zeros_count, sdata[threadIdx.x]);
}
```

프로덕션 코드에서는 CUB의 기본 요소를 사용하여 전체 4비트 radix sort를 구현합니다:

```c
#include <cub/cub.cuh>

void radix_sort_example(uint32_t *d_keys, int n) {
    // CUB DeviceRadixSort — 고도로 최적화된 4비트 radix sort
    cub::DoubleBuffer<uint32_t> d_keys_buf(d_keys, d_tmp_keys);

    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys_buf, n);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys_buf, n);

    // 결과가 대체 버퍼에 있으면 복사
    if (d_keys_buf.Current() != d_keys)
        cudaMemcpy(d_keys, d_keys_buf.Current(), n * sizeof(uint32_t),
                   cudaMemcpyDeviceToDevice);

    cudaFree(d_temp);
}

// key-value 쌍 정렬 (예: int 인덱스가 있는 float key)
void radix_sort_pairs(float *d_keys, int *d_vals, int n) {
    cub::DoubleBuffer<float> d_keys_buf(d_keys, d_tmp_keys);
    cub::DoubleBuffer<int>   d_vals_buf(d_vals, d_tmp_vals);

    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_buf, d_vals_buf, n);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_buf, d_vals_buf, n);
    cudaFree(d_temp);
}
```

---

## 4. GPU에서의 Merge Sort

GPU merge sort는 두 단계로 작동합니다:
1. **로컬 정렬**: 각 block이 부배열을 정렬 (bitonic 또는 bitonic+shared 사용)
2. **글로벌 병합**: 정렬된 런을 반복적으로 병합 (stride 두 배씩 증가)

```c
// 두 포인터를 사용해 두 정렬된 절반을 병합 (thread당 순차)
__global__ void merge_step(const float *in, float *out, int width, int n) {
    // 각 thread block이 하나의 병합된 세그먼트를 처리
    int seg_start = blockIdx.x * (2 * width);
    int mid       = min(seg_start + width, n);
    int seg_end   = min(seg_start + 2 * width, n);

    // 간단한 직렬 병합 (개선된 버전은 병렬 병합 사용)
    int l = seg_start, r = mid, out_i = seg_start;
    while (l < mid && r < seg_end) {
        if (in[l] <= in[r]) out[out_i++] = in[l++];
        else                 out[out_i++] = in[r++];
    }
    while (l < mid)     out[out_i++] = in[l++];
    while (r < seg_end) out[out_i++] = in[r++];
}

void merge_sort(float *d_data, int n) {
    float *d_tmp;
    cudaMalloc(&d_tmp, n * sizeof(float));

    // Phase 1: 각 BLOCK 요소 block을 로컬 정렬
    const int BLOCK = 1024;
    bitonic_sort_shared<<<(n + BLOCK - 1) / BLOCK, BLOCK,
                          BLOCK * sizeof(float)>>>(d_data, n);

    // Phase 2: 병합 pass
    float *src = d_data, *dst = d_tmp;
    for (int width = BLOCK; width < n; width <<= 1) {
        int num_segs = (n + 2 * width - 1) / (2 * width);
        merge_step<<<num_segs, 1>>>(src, dst, width, n);
        cudaDeviceSynchronize();
        float *swap = src; src = dst; dst = swap;
    }

    if (src != d_data) cudaMemcpy(d_data, src, n * sizeof(float),
                                  cudaMemcpyDeviceToDevice);
    cudaFree(d_tmp);
}
```

GPU merge sort는 비교 함수가 비싼 경우(예: 문자열이나 복합 key 정렬) 유용합니다. merge sort는 radix sort의 고정 자릿수 분해 pass에 비해 O(n log n) 비교만 필요하기 때문입니다.

---

## 5. Thrust Sort

Thrust는 가장 높은 수준의 인터페이스를 제공합니다 — 코드 한 줄:

```c
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

void thrust_sort_example() {
    // 편의: thrust::device_vector가 GPU 메모리를 관리
    thrust::device_vector<float> d_vec = {3.f, 1.f, 4.f, 1.f, 5.f, 9.f};
    thrust::sort(d_vec.begin(), d_vec.end());       // 오름차순
    thrust::sort(d_vec.begin(), d_vec.end(),
                 thrust::greater<float>());          // 내림차순

    // 커스텀 비교자로 정렬
    thrust::sort(thrust::device, d_vec.begin(), d_vec.end(),
                 [] __device__ (float a, float b) { return fabsf(a) < fabsf(b); });

    // key-value 쌍 정렬 (argsort 등가)
    thrust::device_vector<int> d_idx(d_vec.size());
    thrust::sequence(d_idx.begin(), d_idx.end());   // 0, 1, 2, ...
    thrust::sort_by_key(d_vec.begin(), d_vec.end(), d_idx.begin());
    // d_idx는 이제 정렬된 순서의 원래 인덱스를 보유

    // 원시 CUDA 포인터와의 상호 운용
    float *d_raw;  int n = 1 << 20;
    cudaMalloc(&d_raw, n * sizeof(float));
    thrust::sort(thrust::device, d_raw, d_raw + n);
    cudaFree(d_raw);
}
```

Thrust의 `sort`는 내부적으로 기본 타입(int, float, double)에는 radix sort를, 복잡한 비교자에는 merge sort를 사용합니다.

---

## 6. 알고리즘 비교 및 선택 가이드

```
알고리즘            복잡도           안정?  최적 사용 사례
-----------------------------------------------------------------
Bitonic sort      O(n log²n)       아니오  n < 100만, 고정 HW,
                                          oblivious 네트워크 필요
Radix sort (CUB)  O(n * k/b)       예     정수, float,
                  (k=키 비트,             큰 n (>100만)
                   b=비트/pass)
Merge sort        O(n log n)       예     복잡한 비교자,
                                          연결된 구조
Thrust::sort      O(n log n)       의존적  가장 빠르게 작성;
                  (자동 선택)             POD 타입에 radix 사용

성능 (N=1억 2800만 int32, RTX 3090, 단일 pass 종단간):
  CUB DeviceRadixSort:  ~4 GB/s 처리량
  Thrust::sort (int):   ~3.8 GB/s (내부적으로 CUB 사용)
  Bitonic:              ~1.2 GB/s (O(log²n) 오버헤드)
```

**선택 규칙:**
- **정수 또는 float, 큰 n**: `cub::DeviceRadixSort` 또는 `thrust::sort` 사용
- **안정 정렬 필요**: `cub::DeviceRadixSort` (안정적) 또는 `thrust::stable_sort` 사용
- **복잡한 커스텀 비교자**: device lambda와 함께 `thrust::sort` 사용
- **작은 n (< 2048) 또는 단일 block**: shared memory에서 bitonic sort
- **가변 길이 key 또는 문자열**: 요소당 비교 kernel을 사용한 merge sort

---

## 핵심 요약

- **Bitonic sort**는 데이터 독립적 (divergence 없음)이지만 O(n log²n) — 작은 n 또는 oblivious 네트워크 요구 사항에 최적
- **Radix sort**는 pass당 O(n) (32비트 key에 8 pass) — 정수 또는 float의 큰 배열에 가장 빠름
- 각 radix pass는: bucket당 카운트 → 배타적 scan → scatter; scan이 병목
- **Thrust::sort**는 기본 타입에 radix sort를 자동으로 선택; 코드 한 줄로 정확하고 빠름
- **CUB DeviceRadixSort**는 가장 세밀한 제어 제공 (key-value 쌍, 부분 비트 범위, 인플레이스 이중 버퍼)
- 정렬 성능은 메모리 대역폭 바운드; 최고의 GPU 정렬은 메모리 대역폭 1바이트당 ~4개 요소를 달성

---

**다음**: [17. Stencil Computations](./17_Stencil_Computations.md) — shared memory tiling, halo cell, 열 방정식을 위한 시간 스텝 루프로 1D/2D/3D stencil kernel을 구현합니다.
