# 15. Parallel Scan / Prefix Sum

**이전**: [Parallel Reduction](./14_Parallel_Reduction.md) | **다음**: [Parallel Sort](./16_Parallel_Sort.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 포괄적(inclusive) prefix sum과 배타적(exclusive) prefix sum을 구별하고 서로 변환하기
2. Hillis-Steele 포괄적 scan 구현하기 (O(n log n) 작업량, O(log n) 깊이)
3. Blelloch 작업 효율적 배타적 scan 구현하기 (O(n) 작업량, O(log n) 깊이)
4. Stream compaction에 scan 적용하기 (0이 아닌 요소 필터링)
5. 프로덕션 품질 scan 연산을 위해 CUB `DeviceScan` 사용하기

---

## 1. Prefix Sum이란?

**포괄적 scan**: output[i] = input[0] + input[1] + ... + input[i]
**배타적 scan**: output[i] = input[0] + input[1] + ... + input[i-1]  (output[0] = 0)

```
입력:              [3,  1,  4,  1,  5,  9,  2,  6]
포괄적 scan:       [3,  4,  8,  9, 14, 23, 25, 31]
배타적 scan:       [0,  3,  4,  8,  9, 14, 23, 25]
```

**관계**: exclusive[i] = inclusive[i-1] (exclusive[0] = 항등원).

Prefix sum은 다음의 기반입니다:
- Stream compaction (술어를 만족하는 요소 선택)
- Radix sort (자릿수 bucket의 오프셋 계산)
- Segmented 연산 (가변 길이 병렬 작업)
- 부하 분산 (불균등한 작업 청크 배분)

---

## 2. Hillis-Steele 포괄적 Scan (O(n log n) 작업량)

Hillis-Steele 알고리즘은 추가 작업 비용(O(n log n) 총 덧셈)으로 최소 깊이(O(log n) 단계)를 달성합니다:

```
Step 1 (stride=1): [3, 1+3, 4+1, 1+4, 5+1, 9+5, 2+9, 6+2]
                 = [3,   4,   5,   5,   6,  14,  11,   8]
Step 2 (stride=2): [3, 4, 5+3, 5+4, 6+5, 14+5, 11+6, 8+14]
                 = [3, 4,   8,   9,  11,   19,   17,   22]
Step 3 (stride=4): [3, 4, 8, 9, 11+3, 19+4, 17+8, 22+9]
                 = [3, 4, 8, 9,   14,   23,   25,   31]  ✓
```

```c
// Hillis-Steele 포괄적 scan (단일 block, n <= blockDim.x)
__global__ void scan_hillis_steele(const float *g_in, float *g_out, int n) {
    extern __shared__ float temp[];  // 이중 버퍼 shared memory

    int tid = threadIdx.x;
    int pout = 0, pin = 1;          // ping-pong 버퍼 인덱스

    // 입력 로드
    temp[pout * n + tid] = (tid < n) ? g_in[tid] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < n; stride <<= 1) {
        pout = 1 - pout;  // 버퍼 교환
        pin  = 1 - pout;

        if (tid >= stride)
            temp[pout * n + tid] = temp[pin * n + tid] + temp[pin * n + tid - stride];
        else
            temp[pout * n + tid] = temp[pin * n + tid];

        __syncthreads();
    }

    if (tid < n) g_out[tid] = temp[pout * n + tid];
}
```

**Hillis-Steele 사용 시기**: 총 작업량보다 깊이(레이턴시)가 더 중요할 때 — 예를 들어 32개 lane이 동기식으로 실행되는 단일 warp 내부.

---

## 3. Blelloch 작업 효율적 배타적 Scan (O(n) 작업량)

Blelloch scan은 O(n) 총 덧셈만 수행하며 순차 복잡도와 일치하고 깊이는 O(log n)입니다. 두 단계를 사용합니다:

**Phase 1 — Up-sweep (reduce)**: 리프에서 루트까지 reduction 트리를 구축합니다.
**Phase 2 — Down-sweep**: 루트에서 리프로 순회하며 부분 합을 분배합니다.

```c
// Blelloch 배타적 scan — 단일 block, n은 2의 거듭제곱이어야 함
__global__ void scan_blelloch(float *g_data, float *g_out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // shared memory에 로드
    sdata[tid]     = (2 * tid     < n) ? g_data[2 * tid]     : 0.0f;
    sdata[tid + 1] = (2 * tid + 1 < n) ? g_data[2 * tid + 1] : 0.0f;
    // (각 thread가 2개 요소를 처리; n/2개 thread 실행)

    // Phase 1: Up-sweep (reduce)
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset <<= 1;
    }

    // 루트를 항등원으로 설정 (합의 경우 0)
    if (tid == 0) sdata[n - 1] = 0.0f;

    // Phase 2: Down-sweep
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float tmp   = sdata[ai];
            sdata[ai]   = sdata[bi];
            sdata[bi]  += tmp;
        }
    }
    __syncthreads();

    // 출력 작성
    if (2 * tid     < n) g_out[2 * tid]     = sdata[tid];
    if (2 * tid + 1 < n) g_out[2 * tid + 1] = sdata[tid + 1];
}
```

**복잡도 비교:**

```
알고리즘        작업량          깊이 (단계)   추가 메모리
--------------------------------------------------------------
순차            O(n)          O(n)            O(1)
Hillis-Steele   O(n log n)    O(log n)        O(n) 이중 버퍼
Blelloch        O(n)          O(log n)        O(n) shared memory
```

작업 효율성이 중요할 만큼 n이 충분히 클 때 Blelloch가 선호됩니다.

---

## 4. Shuffle을 이용한 Warp 수준 포괄적 Scan

n ≤ 32의 경우, 가능한 가장 낮은 레이턴시 scan을 위해 shuffle을 사용하세요:

```c
// shuffle up을 사용한 포괄적 warp scan
__device__ float warp_scan_inclusive(float val) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        float y = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) val += y;
    }
    return val;
}

// Block 수준 포괄적 scan: warp scan -> warp 합 결합 -> prefix 추가
__global__ void scan_block_shuffle(const float *g_in, float *g_out, int n) {
    extern __shared__ float warp_sums[];  // warp당 float 하나

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    float val = (tid < n) ? g_in[tid] : 0.0f;

    // Step 1: 각 warp 내에서 포괄적 scan
    val = warp_scan_inclusive(val);

    // Step 2: 각 warp의 총합 저장 (lane 31의 값)
    if (lane == 31) warp_sums[wid] = val;
    __syncthreads();

    // Step 3: warp 총합들을 scan (첫 번째 warp만 수행)
    if (wid == 0) {
        float ws = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        ws = warp_scan_inclusive(ws);
        warp_sums[lane] = ws;
    }
    __syncthreads();

    // Step 4: 앞선 warp들의 prefix 추가
    float prefix = (wid > 0) ? warp_sums[wid - 1] : 0.0f;
    val += prefix;

    if (tid < n) g_out[tid] = val;
}
```

---

## 5. 멀티-Block Scan (대형 배열)

하나의 block보다 큰 배열을 scan하려면 block 간 부분 합을 전달해야 합니다. 표준 접근법은 **세 kernel** 전략입니다:

```c
// Kernel 1: 각 block을 독립적으로 scan하고 block 총합 작성
__global__ void scan_blocks(const float *in, float *out, float *block_sums, int n);

// Kernel 2: block_sums 배열 scan (소규모 — block당 하나의 요소)
__global__ void scan_block_sums(float *block_sums, int num_blocks);

// Kernel 3: 스캔된 block prefix를 각 요소에 추가
__global__ void add_block_prefix(float *out, const float *block_sums, int n);

// 호스트 조정
void scan_large(const float *d_in, float *d_out, int n) {
    const int BLOCK = 256;
    int num_blocks = (n + BLOCK - 1) / BLOCK;

    float *d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));

    scan_blocks<<<num_blocks, BLOCK, BLOCK * sizeof(float)>>>(
        d_in, d_out, d_block_sums, n);

    // 재귀적으로 block 합 scan (소규모 배열, 단일 block)
    scan_block_sums<<<1, num_blocks, num_blocks * sizeof(float)>>>(
        d_block_sums, num_blocks);

    add_block_prefix<<<num_blocks, BLOCK>>>(d_out, d_block_sums, n);

    cudaFree(d_block_sums);
}
```

현대적 대안: **look-back scan** (CUB 사용) — block들이 전역 동기화 배리어 없이 원자적 플래그를 사용하여 이전 block의 prefix가 사용 가능한지 확인함으로써 진행합니다.

---

## 6. Scan을 이용한 Stream Compaction

Stream compaction은 술어를 만족하는 요소를 선택하여 연속적인 출력 배열로 압축합니다 — GPU 렌더러, 충돌 감지, 그래프 BFS의 핵심 구성 요소입니다:

```c
// 예시: d_in에서 0이 아닌 요소를 d_out으로 압축
// 선택된 요소의 수를 반환합니다.

__global__ void mark_nonzero(const float *d_in, int *d_flags, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_flags[i] = (d_in[i] != 0.0f) ? 1 : 0;
}

__global__ void scatter(const float *d_in, const int *d_flags,
                        const int *d_scan,  // d_flags의 배타적 scan
                        float *d_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_flags[i]) {
        d_out[d_scan[i]] = d_in[i];  // scan된 위치로 scatter
    }
}

int stream_compact(const float *d_in, float *d_out, int n) {
    const int BLOCK = 256;
    int *d_flags, *d_scan;
    cudaMalloc(&d_flags, n * sizeof(int));
    cudaMalloc(&d_scan,  n * sizeof(int));

    // Step 1: 마킹
    mark_nonzero<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_flags, n);

    // Step 2: 출력 위치를 구하기 위해 플래그의 배타적 scan
    // (실제로는 CUB DeviceScan::ExclusiveSum 사용)
    exclusive_scan(d_flags, d_scan, n);

    // Step 3: 선택된 요소들 scatter
    scatter<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_flags, d_scan, d_out, n);

    // 출력 요소 수 = scan[n-1] + flags[n-1]
    int last_flag, last_scan;
    cudaMemcpy(&last_flag, d_flags + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_scan, d_scan  + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_flags); cudaFree(d_scan);
    return last_flag + last_scan;
}
```

---

## 7. CUB DeviceScan 사용하기

```c
#include <cub/cub.cuh>

void cub_exclusive_scan(const float *d_in, float *d_out, int n) {
    void   *d_temp = nullptr;
    size_t  temp_bytes = 0;

    // 임시 저장소 크기 조회
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, n);

    // 할당
    cudaMalloc(&d_temp, temp_bytes);

    // 실행
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
}

// 포괄적 scan
void cub_inclusive_scan(const int *d_in, int *d_out, int n) {
    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out, n);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out, n);
    cudaFree(d_temp);
}

// 세그먼트 scan: 플래그로 정의된 세그먼트 내에서 독립적인 scan
// cub::DeviceScan::ExclusiveSumByKey(d_keys, d_vals, d_out, n)
```

CUB의 DeviceScan은 **decoupled look-back** 알고리즘을 사용합니다: block들이 전역 동기화 배리어를 기다리는 대신 원자적 플래그를 통해 이전 block의 prefix가 사용 가능해지는 즉시 진행합니다. 이를 통해 peak 메모리 대역폭에 근접합니다.

---

## 핵심 요약

- **포괄적 scan**: output[i]는 input[i]를 포함; **배타적 scan**: output[i]는 input[i]를 제외 (output[0] = 0)
- Hillis-Steele은 O(log n) 깊이를 위해 O(n log n) 작업량 사용 — 소규모 warp 수준 scan에 적합
- **Blelloch**는 up-sweep/down-sweep을 통해 O(n) 작업량에 O(log n) 깊이를 달성 — 큰 n에 선호
- Warp shuffle (`__shfl_up_sync`)은 shared memory 없이 가장 빠른 warp 수준 scan을 제공
- **Stream compaction** = 마킹 + 배타적 scan + scatter — 데이터 의존적 출력 크기를 가능하게 하는 3단계 파이프라인
- **CUB `DeviceScan`**은 near-peak 대역폭을 위해 decoupled look-back을 사용; 프로덕션에서 사용하세요

---

**다음**: [16. Parallel Sort](./16_Parallel_Sort.md) — bitonic sort, radix sort, thrust::sort를 구현하고 GPU에서 각 알고리즘을 언제 사용해야 하는지 이해합니다.
