# 14. Parallel Reduction

**이전**: [CUDA Graphs](./13_CUDA_Graphs.md) | **다음**: [Parallel Scan Prefix Sum](./15_Parallel_Scan_Prefix_Sum.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 트리 reduction 패턴과 O(log N) 깊이 복잡도 설명하기
2. 나이브 reduction kernel에서 warp divergence 식별 및 제거하기
3. `__shfl_down_sync`를 사용한 warp shuffle reduction 구현하기
4. 임의로 큰 배열에 대한 다단계 device-level reduction 설계하기
5. 프로덕션 품질 대안으로 CUB `DeviceReduce::Sum` 사용하기

---

## 1. Reduction이 근본적인 이유

**Reduction**은 결합 연산자(합, 최대, 최소, 곱)를 사용해 배열에서 단일 스칼라를 계산합니다. GPU 컴퓨팅 전반에 걸쳐 등장합니다:

```
입력:  [3, 1, 4, 1, 5, 9, 2, 6]
출력: 31                          (합 reduction)
출력: 9                           (최대 reduction)
```

Reduction은 GPU의 정식 "다-대-하나" 패턴입니다. 이를 마스터하면 warp 수준 프로그래밍, shared memory 동기화, 분기 비용을 배울 수 있으며, 이 기술들은 scan, sort, histogram kernel에 직접 적용됩니다.

---

## 2. 나이브 트리 Reduction (인터리브 어드레싱)

교과서적 접근법은 각 단계마다 활성 thread를 2로 나눕니다:

```c
// 나이브 reduction — 인터리브 어드레싱이 warp divergence를 유발함
__global__ void reduce_naive(const float *g_in, float *g_out, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    // 각 단계에서 활성 thread 수를 절반으로 줄임
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tid % (2 * stride) == 0) {           // 절반의 thread가 유휴
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```

**문제 — warp divergence**: `tid % (2 * stride) == 0` 조건으로 인해 모든 warp에서 절반의 thread가 다른 분기를 취합니다. stride=1일 때 32-thread warp당 16개 thread가 유휴 상태입니다. stride=16일 때 warp당 31개 thread가 유휴 상태입니다. 32개 thread 모두가 여전히 두 경로를 실행하므로 issue slot이 낭비됩니다.

---

## 3. Divergence-Free Reduction (순차 어드레싱)

모듈식 인덱싱을 순차 어드레싱으로 교체하여 모든 활성 thread가 연속적으로 위치하도록 해, warp 내 divergence를 제거합니다:

```c
// Divergence-free reduction — 순차 어드레싱
__global__ void reduce_sequential(const float *g_in, float *g_out, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    // stride는 block의 절반부터 시작, 하위 절반의 thread가 항상 활성
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {                      // warp 내 divergence 없음
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```

**추가 최적화: global load 시 첫 번째 덧셈.** 각 thread가 두 개의 요소를 로드하고 shared memory에 저장하기 전에 합산합니다. 이렇게 하면 필요한 block 수가 절반으로 줄고 메모리 접근당 산술 연산이 두 배가 됩니다:

```c
__global__ void reduce_load2(const float *g_in, float *g_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (i < n)              val  = g_in[i];
    if (i + blockDim.x < n) val += g_in[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```

---

## 4. Shuffle을 이용한 Warp 수준 Reduction

마지막 32개 thread(하나의 warp)에서는 `__syncthreads()`가 불필요합니다 — 동일한 warp의 thread들은 항상 동기화되어 있습니다. 더 나은 방법: **warp shuffle**을 사용하여 shared memory를 우회하고 register에서 값을 교환합니다:

```c
// shuffle down을 사용한 warp 수준 reduction
__device__ float warp_reduce_sum(float val) {
    // 전체 마스크: 32개 lane 모두 참여
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // lane 0이 warp 합계를 보유
}

__global__ void reduce_warp_shuffle(const float *g_in, float *g_out, int n) {
    extern __shared__ float warp_sums[];

    unsigned int tid  = threadIdx.x;
    unsigned int lane = tid & 31;            // warp 내 lane
    unsigned int wid  = tid >> 5;            // block 내 warp 인덱스
    unsigned int i    = blockIdx.x * (blockDim.x * 2) + tid;

    // 두 요소를 로드하고 합산
    float val = 0.0f;
    if (i < n)              val  = g_in[i];
    if (i + blockDim.x < n) val += g_in[i + blockDim.x];

    // Step 1: 각 warp 내에서 reduction (shared memory 불필요)
    val = warp_reduce_sum(val);

    // Step 2: 각 warp의 lane 0이 합계를 shared memory에 씀
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();

    // Step 3: 첫 번째 warp가 warp 합계들을 reduction
    val = (tid < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    if (tid == 0) g_out[blockIdx.x] = val;
}
```

**warp tail에서 `__shfl_down_sync`가 shared memory보다 나은 이유:**
- Register 간 전송: 1–2 사이클
- Shared memory: 20–30 사이클 (bank conflict 없는 경우) 또는 그 이상
- `__syncwarp()` 불필요 — shuffle은 warp 내에서 이미 동기화됨

---

## 5. 다단계 Device 수준 Reduction

단일 kernel 호출은 `N`개 값을 `gridDim.x`개 부분 합으로만 줄일 수 있습니다. 큰 배열의 경우 두 번째(또는 재귀적) reduction을 실행합니다:

```c
// 호스트 측 다단계 reduction
float device_reduce_sum(const float *d_in, int n) {
    const int BLOCK = 256;
    int grid  = (n + BLOCK * 2 - 1) / (BLOCK * 2);  // "load 2" block
    int smem  = (BLOCK / 32) * sizeof(float);         // warp sum 저장 공간

    float *d_partial;
    cudaMalloc(&d_partial, grid * sizeof(float));

    // Stage 1: N → grid개 부분 합으로 reduction
    reduce_warp_shuffle<<<grid, BLOCK, smem>>>(d_in, d_partial, n);

    float result;
    if (grid == 1) {
        // 완료 — 단일 결과 복사
        cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        // Stage 2: 부분 합 reduction (재귀)
        result = device_reduce_sum(d_partial, grid);
    }

    cudaFree(d_partial);
    return result;
}
```

프로덕션 코드에서는 호스트 측 재귀를 피하세요. 대신 atomicAdd를 사용해 부분 합을 누적하는 두 번째 고정 kernel을 실행합니다:

```c
__global__ void reduce_atomic_final(const float *partials, float *result, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? partials[i] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);  // 안전한 멀티-block 누적
}
```

---

## 6. CUB DeviceReduce::Sum 사용하기

프로덕션 코드에서는 NVIDIA의 CUB 라이브러리를 사용하세요 — 직접 작성한 kernel보다 성능이 뛰어난 수동 튜닝 구현이 포함되어 있습니다:

```c
#include <cub/cub.cuh>

void cub_reduce_example(const float *d_in, float *d_out, int n) {
    // Step 1: 임시 저장소 크기 조회
    void   *d_temp = nullptr;
    size_t  temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);

    // Step 2: 임시 저장소 할당
    cudaMalloc(&d_temp, temp_bytes);

    // Step 3: reduction 실행 (단일 API 호출)
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
}

// CUB는 다음도 지원합니다: Min, Max, ArgMin, ArgMax, Reduce (커스텀 연산)
// 이진 연산자를 사용한 커스텀 reduction:
struct MaxAbsOp {
    __device__ float operator()(float a, float b) {
        return fmaxf(fabsf(a), fabsf(b));
    }
};

void cub_max_abs(const float *d_in, float *d_out, int n) {
    void *d_temp = nullptr; size_t temp_bytes = 0;
    MaxAbsOp op;
    cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out, n, op, 0.0f);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out, n, op, 0.0f);
    cudaFree(d_temp);
}
```

**성능 비교 (N = 1억 2800만 float, RTX 3090):**

```
Kernel                       시간 (ms)    메모리 대역폭 %
-------------------------------------------------------
Naive (인터리브)              8.4          24%
Sequential (divergence-free) 4.1          49%
Warp shuffle                 2.2          91%
CUB DeviceReduce::Sum        2.1          95%
이론적 최대 (BW 한계)         ~2.0 ms     100%
```

Reduction은 **메모리 대역폭 바운드**입니다 — 최적 구현은 산술이 아닌 global memory에서 데이터를 읽는 속도에 의해 제한됩니다.

---

## 7. 다른 연산자에 대한 Reduction

동일한 warp shuffle 패턴은 결합적이고 교환적인 모든 연산자에 일반화됩니다:

```c
// 최대 reduction
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// ArgMax (인덱스 + 값 함께)
struct ArgMax { float val; int idx; };
__device__ ArgMax warp_reduce_argmax(ArgMax a) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, a.val, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, a.idx, offset);
        if (other_val > a.val) { a.val = other_val; a.idx = other_idx; }
    }
    return a;
}

// 내적 (쌍별 곱셈 후 reduction)
__global__ void dot_product(const float *a, const float *b, float *out, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? a[i] * b[i] : 0.0f;
    val = warp_reduce_sum(val);
    // ... (이전과 동일한 warp -> block -> global 패턴)
}
```

---

## 핵심 요약

- 나이브 인터리브 어드레싱 reduction은 divergence로 인해 warp slot을 낭비합니다; 순차 어드레싱으로 해결됩니다
- **Warp shuffle** (`__shfl_down_sync`)은 마지막 warp에서 shared memory 왕복을 제거하여 레이턴시를 단계당 1–2 사이클로 줄입니다
- Reduction은 **메모리 대역폭 바운드** — 최적 kernel은 peak device 대역폭의 ~95%에서 실행됩니다
- 다단계 reduction: Stage 1은 N개 요소를 `gridDim.x`개 부분 합으로 줄이고; Stage 2(또는 atomic)가 완료합니다
- **CUB `DeviceReduce::Sum`**은 프로덕션 선택입니다: 모든 엣지 케이스를 자동으로 처리하고 peak 대역폭에 근접합니다
- Warp shuffle 패턴은 모든 결합 연산자에 일반화됩니다: max, min, 내적, argmax

---

**다음**: [15. Parallel Scan Prefix Sum](./15_Parallel_Scan_Prefix_Sum.md) — 포괄적(inclusive) 및 배타적(exclusive) prefix sum을 구축합니다. stream compaction, radix sort, segmented 연산의 핵심 기본 요소입니다.
