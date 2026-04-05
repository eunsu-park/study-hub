# 18. Histogram과 Binning

**이전**: [Stencil Computations](./17_Stencil_Computations.md) | **다음**: [Sparse Matrix Ops](./19_Sparse_Matrix_Ops.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 전역 atomic histogram 구현 및 직렬화 병목 이해하기
2. shared memory 사분할(privatization)로 atomic 경쟁을 ~(block 크기 / bin 수) 배 줄이기
3. 8-bit accumulator와 overflow 처리를 사용한 256-bin histogram 최적화하기
4. 결합 확률 추정을 위한 2D histogram 계산하기
5. CUB `DeviceHistogram` 사용 및 atomic 기반 vs sort 기반 방식 선택하기

---

## 1. GPU에서 Histogram이 어려운 이유

Histogram은 입력 값이 각 bin에 몇 개 속하는지 집계합니다. 개념적 업데이트인 `bins[bucket(x)]++`는 데이터에 의해 결정된 주소에 대한 **읽기-수정-쓰기** 연산입니다 — 여러 thread가 동일한 bin에 매핑되면 쓰기 충돌이 발생합니다.

```
입력: [2, 5, 2, 7, 2, 5, 0, 2]  (8개 요소, 8개 bin)
결과: [1, 0, 4, 0, 0, 2, 0, 1]

문제: thread 0,2,4,6이 동시에 bin[2]를 증가시키려 하면,
atomic 없이는 세 개의 증가가 유실됩니다.
```

높은 병렬성을 유지하면서 쓰기 충돌을 올바르게 해결하는 것이 과제입니다.

---

## 2. 기본: 전역 Atomic Histogram

```c
// 가장 단순한 올바른 구현 — thread당 하나의 atomic
__global__ void histogram_global_atomic(
    const int *data, int *hist, int n, int num_bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = data[i];           // 데이터가 이미 [0, num_bins) 범위라고 가정
        atomicAdd(&hist[bin], 1);    // 전역 atomic
    }
}

// 호스트
void run_histogram_global(const int *d_data, int *d_hist, int n, int B) {
    cudaMemset(d_hist, 0, B * sizeof(int));
    const int BLOCK = 256;
    histogram_global_atomic<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        d_data, d_hist, n, B);
}
```

**성능 문제**: B=256 bin, N=1억 2800만 요소일 때 각 bin은 평균 N/B = 50만 번 증가됩니다. 피크 시 ~4096개 thread가 256개 주소에 동시에 `atomicAdd`를 발행하므로 전역 L2 캐시에서 심각한 직렬화가 발생합니다. 처리량: ~5000만 요소/초 vs 10TB/s 메모리 대역폭 (이론값보다 200× 느림).

---

## 3. Shared Memory 사분할(Privatization)

각 block이 shared memory에 histogram의 **비공개 복사본**을 유지합니다. block의 데이터를 처리한 후, 비공개 histogram을 block당 bin당 하나의 atomic으로 전역 histogram에 병합합니다:

```c
// Shared memory 사분할 histogram
__global__ void histogram_smem(
    const int *data, int *hist, int n, int num_bins)
{
    extern __shared__ int local_hist[];  // 크기 = num_bins * sizeof(int)

    int tid = threadIdx.x;

    // 로컬 histogram을 0으로 초기화
    for (int b = tid; b < num_bins; b += blockDim.x)
        local_hist[b] = 0;
    __syncthreads();

    // 각 thread가 여러 요소를 처리 (grid-stride loop)
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride)
        atomicAdd(&local_hist[data[i]], 1);  // shared mem atomic (빠름)
    __syncthreads();

    // 로컬 histogram을 전역 histogram에 병합
    for (int b = tid; b < num_bins; b += blockDim.x)
        atomicAdd(&hist[b], local_hist[b]);  // block당 bin당 하나의 전역 atomic
}

void run_histogram_smem(const int *d_data, int *d_hist, int n, int B) {
    cudaMemset(d_hist, 0, B * sizeof(int));
    const int BLOCK = 256;
    // shared mem 재사용률을 높이기 위해 grid 크기 제한 (각 block이 많은 요소 처리)
    int grid = min((n + BLOCK - 1) / BLOCK, 1024);
    histogram_smem<<<grid, BLOCK, B * sizeof(int)>>>(d_data, d_hist, n, B);
}
```

**Shared memory atomic**은 전역 메모리 atomic보다 약 10–30배 빠릅니다(SM의 L1 캐시 내에서 동작). 각 shared atomic 충돌은 ~4 사이클에 해결되는 반면 전역은 ~100+ 사이클이 걸립니다.

**메모리 제약**: 256-bin histogram은 256 × 4 = 1024바이트의 shared memory를 사용하므로 매우 작습니다. 4096-bin histogram은 16KB를 사용하며 48KB shared 내에 여전히 들어맞습니다. ~8192 bin을 초과하면 shared memory 사분할이 비실용적이 됩니다.

---

## 4. 8-bit Accumulator를 사용한 256-Bin 최적화

256 bin에 대해 bin당 `uint8_t` (4배 작음)를 사용하고 bin이 255에 도달하면 전역으로 플러시합니다:

```c
// overflow 검사가 있는 8-bit 사분할 histogram
__global__ void histogram_256_u8(
    const uint8_t *data, int *hist, int n)
{
    // 256 bin × 1바이트 = 256바이트의 shared mem
    __shared__ uint8_t local8[256];
    int tid = threadIdx.x;

    // 64-thread 저장(각 4바이트)을 사용하여 256바이트를 0으로 초기화
    if (tid < 64) ((uint32_t*)local8)[tid] = 0;
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        int bin = data[i];
        // 증가; overflow 시 전역으로 플러시
        if (++local8[bin] == 0) {          // 순환 발생 (255 → 0)
            atomicAdd(&hist[bin], 256);    // 순환된 256을 복구
        }
    }
    __syncthreads();

    // 남은 카운트 플러시
    if (tid < 256) atomicAdd(&hist[tid], local8[tid]);
}
```

이렇게 하면 shared memory 사용량이 절반으로 줄고, 일부 GPU 아키텍처에서 여러 thread가 동일한 bin을 동시에 업데이트할 때 8-bit 워드가 32-bit 워드보다 atomic 업데이트 비용이 낮아 warp 수준 처리량이 향상될 수 있습니다.

---

## 5. 2D Histogram (결합 분포)

2D histogram은 쌍 (x[i], y[i]) → bin (bx, by)을 집계하며, 공발생 행렬, 결합 확률 추정, 색상 histogram 디스크립터에 유용합니다:

```c
// 2D histogram: Bx bin × By bin 격자
__global__ void histogram_2d(
    const float *x_data, const float *y_data, int *hist,
    int n, int Bx, int By,
    float x_min, float x_max, float y_min, float y_max)
{
    extern __shared__ int local_hist[];  // Bx * By 개 int

    int tid = threadIdx.x;
    int total_bins = Bx * By;

    for (int b = tid; b < total_bins; b += blockDim.x) local_hist[b] = 0;
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        float x = x_data[i], y = y_data[i];
        if (x < x_min || x >= x_max || y < y_min || y >= y_max) continue;

        int bx = (int)((x - x_min) / (x_max - x_min) * Bx);
        int by = (int)((y - y_min) / (y_max - y_min) * By);
        bx = min(bx, Bx - 1);
        by = min(by, By - 1);

        atomicAdd(&local_hist[by * Bx + bx], 1);
    }
    __syncthreads();

    for (int b = tid; b < total_bins; b += blockDim.x)
        atomicAdd(&hist[b], local_hist[b]);
}
```

대형 2D histogram (예: 1024×1024 = 400만 bin)의 경우 사분할이 shared memory에 더 이상 맞지 않습니다. **sort 기반 방식**으로 전환하세요: 모든 (bx, by) 쌍을 선형화된 bin 인덱스로 정렬한 후 run-length encoding으로 집계합니다.

---

## 6. CUB DeviceHistogram 사용하기

CUB는 일반적인 사용 사례에 대한 최적화된 histogram 구현을 제공합니다:

```c
#include <cub/cub.cuh>

// 단일 채널 histogram (예: 그레이스케일 이미지)
void cub_histogram_single(const uint8_t *d_samples, int *d_hist,
                           int n_samples) {
    const int NUM_BINS = 256;
    int lower = 0, upper = 256;  // 샘플 범위 [lower, upper)

    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        d_samples, d_hist, NUM_BINS + 1,  // +1: CUB는 bin 경계를 집계
        lower, upper, n_samples);

    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        d_samples, d_hist, NUM_BINS + 1,
        lower, upper, n_samples);

    cudaFree(d_temp);
}

// 다중 채널 histogram (예: RGB 이미지 — 3채널, 각 256 bin)
void cub_histogram_multi_channel(const uint8_t *d_image,   // 인터리브 RGB
                                  int *d_hist_r, int *d_hist_g, int *d_hist_b,
                                  int n_pixels) {
    const int NUM_CHANNELS = 3;
    const int NUM_ACTIVE   = 3;   // 모든 채널 활성
    const int NUM_BINS     = 256;

    int* d_hists[3] = {d_hist_r, d_hist_g, d_hist_b};
    int  levels[3]  = {NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1};
    int  lower[3]   = {0, 0, 0};
    int  upper[3]   = {256, 256, 256};

    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE>(
        d_temp, temp_bytes,
        d_image, d_hists, levels, lower, upper, n_pixels);

    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE>(
        d_temp, temp_bytes,
        d_image, d_hists, levels, lower, upper, n_pixels);

    cudaFree(d_temp);
}
```

---

## 7. Atomic vs Sort 기반 Histogram

```
방식              사용 시기                              처리량
-----------------------------------------------------------------------
전역 atomic       빠른 프로토타입, B가 크고,             ~5000만 요소/초
                  불균일 분포                            (경쟁 제한)

Shared atomic     B ≤ 8192, 균일/보통 분포,             ~5억 요소/초
(사분할)          범용으로 최선의 선택

CUB Histogram     B가 2의 거듭제곱, 256–4096 bin,       ~20억 요소/초
                  프로덕션 코드

Sort 기반         매우 큰 B (>16K bin), 정확한 bucket   ~10억 요소/초
                  위치 필요, 이후 bucket별 처리          (sort 주도)
```

**Sort 기반 파이프라인**: 키 정렬 → 연속된 동일 키 집계 (run-length encoding). 이 방식은 집계와 함께 출력 위치를 자연스럽게 생성하여 이후 bucket별 작업을 가능하게 합니다.

---

## 핵심 요약

- **전역 atomic**은 올바르지만 L2 경쟁으로 인해 느립니다 — 프로덕션 histogram에는 사용하지 마세요
- **Shared memory 사분할**은 전역 atomic 트래픽을 N에서 `grid × B` atomic으로 줄여 일반적으로 전역 atomic을 1000배 줄입니다
- 256-bin histogram은 shared memory에 편안하게 들어맞습니다(1KB); >8192 bin은 일반적인 shared memory 예산을 초과합니다
- **CUB DeviceHistogram**은 엣지 케이스(범위 외 샘플, 2의 거듭제곱이 아닌 bin, 다중 채널)를 처리하며 대역폭 한계에 근접한 처리량을 달성합니다
- **2D histogram**에서 B×B 격자가 크면 shared memory atomic 대신 sort 기반 방식을 사용해야 합니다
- 작은 B에는 atomic 기반, 큰 B이거나 bucket별 후속 작업이 필요할 때는 sort 기반을 선택하세요

---

**다음**: [19. Sparse Matrix Ops](./19_Sparse_Matrix_Ops.md) — COO, CSR, CSC 형식으로 희소 행렬을 표현하고 cuSPARSE로 효율적인 SpMV를 구현합니다.
