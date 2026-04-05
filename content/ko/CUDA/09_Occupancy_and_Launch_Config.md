# 09. 점유율과 실행 구성

**이전**: [메모리 합치기](./08_Memory_Coalescing.md) | **다음**: [루프라인 모델](./10_Roofline_Model.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 레지스터 수, 공유 메모리, 블록 크기로 점유율 계산
2. 주어진 커널에서 어느 리소스가 점유율을 제한하는지 식별
3. 레지스터 할당 제어를 위한 `__launch_bounds__` 적용
4. CUDA 점유율 계산기 API와 Nsight Compute 사용
5. 높은 점유율이 성능을 향상시키지 않는 경우 이해

---

## 1. 점유율 복습

**점유율** = SM당 활성 warp / SM당 최대 warp

Ampere (A100/RTX 3090): SM당 최대 64 warp (2048 스레드).

점유율을 제한하는 세 가지 리소스:
1. **레지스터** (SM당 64K)
2. **공유 메모리** (SM당 48–228 KB, 구성 가능)
3. **블록 크기** (전체 블록을 할당해야 함)

세 한도 중 **최솟값**이 적용됩니다.

---

## 2. 레지스터 제한 점유율

공식: `max_threads = 64K 레지스터 / 스레드당 레지스터`

```
스레드당 레지스터  →  SM당 최대 스레드  →  SM당 최대 warp  →  점유율
       16                4096                128               100%  (SM 최대 64 warp 초과)
       32                2048                 64               100%  (정확히 SM 최대)
       48                1365                 42                66%
       64                1024                 32                50%
       96                 682                 21                33%
      128                 512                 16                25%
      256                 256                  8                12%
```

**레지스터 사용량 확인 방법**:

```bash
# 방법 1: 컴파일러 상세 출력
nvcc -Xptxas -v mykernel.cu
# 출력: used 32 registers, ...

# 방법 2: Nsight Compute
ncu --metrics launch__registers_per_thread ./my_kernel
```

---

## 3. 공유 메모리 제한 점유율

각 블록이 공유 메모리를 소비합니다. 블록 수가 적을수록 → warp 수 감소 → 점유율 저하.

```
A100: SM당 공유 메모리 228 KB (완전히 구성된 경우)

블록이 공유 메모리 48 KB 사용, 블록에 256 스레드 (8 warp)인 경우:
  SM당 최대 블록 = floor(228 KB / 48 KB) = 4 블록
  활성 warp      = 4 × 8 = 32 warp
  점유율          = 32/64 = 50%

블록이 공유 메모리 8 KB 사용인 경우:
  SM당 최대 블록 = floor(228 KB / 8 KB) = 28, SM 블록 한도 (32)에 의해 제한
  활성 warp      = min(28, 32) × 8 = 224 warp → 최대 64 warp에 의해 제한
  점유율          = 64/64 = 100%
```

**공유 메모리 대 L1 비율**은 Ampere에서 구성 가능합니다:

```c
// 특정 커널에 대한 공유 메모리 크기 설정
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    96 * 1024);  // 이 커널에 96 KB

// 시스템 전체 설정
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);  // 8바이트 뱅크
```

---

## 4. 블록 크기 제한 점유율

SM은 전체 블록을 할당해야 합니다. 블록이 너무 크면 점유율이 저하됩니다:

```
SM 최대: 64 warp (2048 스레드), 최대 32 블록

블록 크기 = 1024 (32 warp):
  최대 블록 = min(2048/1024, 32) = 2 블록
  활성 warp = 2 × 32 = 64 → 100% 점유율 ✓

블록 크기 = 1536 (48 warp):
  최대 블록 = floor(2048/1536) = 1 블록
  활성 warp = 1 × 48 = 48 → 75% 점유율 ✗
  (1536은 2048의 약수가 아님 → 1 블록이 512 스레드 슬롯 낭비)

블록 크기 = 96 (3 warp, 32의 배수이지만 2의 거듭제곱이 아닌 크기):
  최대 블록 = min(2048/96, 32) = 21 블록
  활성 warp = 21 × 3 = 63 → 98% 점유율
  (96 = 3×32은 유효한 warp 정렬 크기; 여기서는 거의 완벽한 점유율이지만,
   큰 블록 수가 스케줄링 오버헤드를 증가시킴)
```

**경험 법칙**: 128, 256, 512 블록 크기는 보편적으로 안전합니다. 192, 320, 384, 448, 640, 768, 896, 1536은 피하세요 — 이들은 종종 warp 슬롯을 낭비합니다.

---

## 5. 점유율 계산기 API

```c
#include <cuda_runtime.h>

// 최적 블록 크기 요청
int minGridSize, optimalBlockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,       // 전체 점유율을 위한 최소 그리드 크기
    &optimalBlockSize,  // 점유율을 최대화하는 블록 크기
    myKernel,           // 커널 함수 포인터
    0,                  // 블록당 동적 공유 메모리 (0 = 없음)
    0                   // 최대 블록 크기 제약 (0 = 제한 없음)
);

// 주어진 블록 크기에 대한 실제 점유율 계산
int activeWarps, maxWarps;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &activeWarps,   // 반환: SM당 활성 블록 (warp/블록 곱해야 함)
    myKernel,
    blockSize,
    sharedMemBytes
);
cudaDeviceGetAttribute(&maxWarps, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
float occupancy = (float)(activeWarps * blockSize / 32) / (maxWarps / 32);
printf("점유율: %.1f%%\n", occupancy * 100);
```

---

## 6. `__launch_bounds__`: 컴파일러 레지스터 힌트

이 커널이 실행될 최대 블록 크기를 컴파일러에 알려 레지스터 할당을 제한합니다:

```c
// 보장: 이 커널은 최대 256 스레드 블록으로만 실행됨
// 그리고 SM당 최소 2 블록 (최소 점유율 힌트)
__global__ __launch_bounds__(256, 2)
void myKernel(float *data, int n) {
    // ...
}
```

효과:
- 컴파일러가 더 적은 레지스터 사용 가능 (점유율 목표 달성)
- 레지스터 스필링 증가 가능 (트레이드오프)
- 컴파일러가 레지스터를 과도하게 할당하고 실행 구성을 알고 있을 때 사용

```c
// 더 공격적: 높은 점유율을 위해 낮은 레지스터 수 강제
__global__ __launch_bounds__(128, 4)  // 최대 128 스레드, SM당 최소 4 블록
void high_occupancy_kernel(float *data) {
    // 컴파일러는 스레드당 최대 64K/(128*4) = 128 레지스터에 맞추려 함
    // 커널이 복잡하면 스필링 발생 가능
}
```

---

## 7. 점유율 vs 성능: 비선형 관계

**높은 점유율이 항상 더 높은 처리량을 의미하지는 않습니다.** 이것이 GPU 최적화에서 가장 중요한 미묘한 점 중 하나입니다.

### 경우 1: 지연 병목 커널 (높은 점유율 도움)

전역 메모리에서 자주 정지하는 커널:
```
- 50% 점유율: 일부 정지를 숨길 수 없음 → GPU 부분적으로 유휴
- 100% 점유율: 모든 정지가 준비된 warp로 커버됨 → 전체 활용
```

### 경우 2: 연산 병목 커널 (점유율 크게 중요하지 않음)

정지 없이 모든 시간을 FP 산술에 소비하는 커널:
```
- 25% 점유율이지만 warp가 절대 정지하지 않음: FP 장치 항상 바쁨 → 동일한 처리량
- warp 추가는 숨길 지연이 없을 때 도움이 되지 않음
```

### 경우 3: 레지스터 집약적 커널 (낮은 점유율이 더 빠를 수 있음)

스레드당 레지스터가 많을수록 = 누산기 더 많음 = 메모리 접근 감소:
```
- L32의 GEMM 커널 (스레드당 레지스터 128개, 25% 점유율)이
  같은 커널 (스레드당 레지스터 32개, 100% 점유율)보다 뛰어남 —
  이유: 레지스터가 많은 버전이 로드된 메모리 바이트당 더 많이 계산하기 때문
```

---

## 8. Nsight Compute로 점유율 프로파일링

```bash
ncu --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__active_cycles_avg,\
    sm__warps_eligible.avg,\
    launch__occupancy_limit_registers,\
    launch__occupancy_limit_shared_mem,\
    launch__occupancy_limit_warps \
    ./my_kernel
```

핵심 메트릭:
- `sm__warps_active.avg.pct_of_peak_sustained_active`: 실제 달성 점유율
- `launch__occupancy_limit_registers`: 레지스터가 바인딩 제약이면 1
- `launch__occupancy_limit_shared_mem`: 공유 메모리가 바인딩 제약이면 1
- `sm__warps_eligible.avg`: 평균적으로 스케줄 준비된 warp 수

`warps_eligible`이 지속적으로 낮으면 실제 점유율 문제가 있습니다. 높지만 처리량이 여전히 낮다면, 병목은 다른 곳에 있습니다 (연산 또는 메모리 대역폭).

---

## 9. 예제: 리덕션 커널 최적화

```c
// 버전 1: 단순 — 공유 메모리 최적화 없음, 스레드당 레지스터 32개
__global__ void reduce_v1(float *in, float *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // ... 단순한 리덕션
}
// 점유율: 100% (레지스터 수가 허용)
// 문제: 자연스러운 warp 전환을 넘어서는 지연 은폐 없음

// 버전 2: warp 셔플, 스레드당 레지스터 20개
__global__ __launch_bounds__(256)
void reduce_v2(float *in, float *out, int n) {
    float val = (blockIdx.x * blockDim.x + threadIdx.x < n) ?
                in[blockIdx.x * blockDim.x + threadIdx.x] : 0.0f;
    val = warp_reduce_sum(val);
    if (threadIdx.x % 32 == 0) out[blockIdx.x * 32 + threadIdx.x / 32] = val;
}
// 점유율: 100%
// 성능: v1보다 3배 빠름 (warp 셔플 + 합치된 접근)
```

---

## 핵심 요약

- 점유율 = min(레지스터 한도, 공유 메모리 한도, 블록 크기 한도)
- 어느 제약이 제한하는지 확인: ncu의 `launch__occupancy_limit_*` 메트릭
- `cudaOccupancyMaxPotentialBlockSize`로 자동으로 최적 블록 크기 찾기
- `__launch_bounds__`는 컴파일러가 레지스터 할당을 줄이도록 안내 — 스필링 발생 가능
- **높은 점유율은 지연 은폐에 도움; 연산 병목 커널에는 도움이 되지 않음**
- 대부분의 커널에서 50%+ 점유율 목표; 높은 레지스터 수를 가진 연산 병목 커널에서는 낮아도 됨

---

**다음**: [10. 루프라인 모델](./10_Roofline_Model.md) — GPU의 루프라인 차트 구성, 산술 강도 계산, 커널이 메모리 병목인지 연산 병목인지 결정.
