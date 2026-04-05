# 05. 공유 메모리와 타일링

**이전**: [CUDA 메모리 모델](./04_CUDA_Memory_Model.md) | **다음**: [Warp 실행과 분기 발산](./06_Warp_Execution_and_Divergence.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 전역 메모리 재사용에 공유 메모리 스테이징이 필요한 이유 설명
2. 올바른 `__syncthreads()` 배치로 타일된 행렬 곱셈 구현
3. 패딩을 이용한 공유 메모리 뱅크 충돌 식별 및 제거
4. Nsight Compute로 공유 메모리 사용량 측정
5. 단순 행렬 곱셈 → 타일된 행렬 곱셈 성능 격차 프로파일링

---

## 1. 재사용 문제

행렬 곱셈 C = A × B (N×N 행렬)를 고려해보세요:

```
원소 C[i][j] = sum_k A[i][k] * B[k][j]
```

단순한 커널에서 스레드 (i, j)는 A에서 행 i의 N개 원소와 B에서 열 j의 N개 원소 모두를 로드합니다 — **출력 원소당 전역 메모리에서 N번의 로드**.

```
N=1024: 각 스레드가 전역 메모리에서 1024 + 1024 = 2048 floats 읽기
총 읽기: N³ = 10억 읽기 (1024×1024 행렬 곱셈)
필요한 전역 메모리 대역폭: 4 GB × ... 달성 가능한 수준을 훨씬 초과
```

핵심 관찰: 같은 블록의 인접 스레드가 행 A[i][*]를 공유합니다. 16개 스레드가 C의 같은 행의 16개 원소를 계산한다면, 각각 A의 같은 행이 필요합니다 — **그 행을 한 번만 공유 메모리에 로드하고 16번 재사용할 수 있습니다**.

---

## 2. 타일링 전략

A, B, C를 `TILE×TILE` 부분행렬로 분할합니다. 각 블록이 C의 타일 하나를 계산합니다:

```
┌───────────────────────────────────────────────┐
│  행렬 A (N×N)      행렬 B (N×N)               │
│                                               │
│  ┌──┬──┬──┬──┐    ┌──┬──┬──┬──┐             │
│  │A₀│A₁│A₂│A₃│    │B₀│  │  │  │             │
│  ├──┼──┼──┼──┤    ├──┼──┼──┼──┤             │
│  │  │  │  │  │    │B₁│  │  │  │             │
│  └──┴──┴──┴──┘    ├──┼──┼──┼──┤             │
│                    │B₂│  │  │  │             │
│  C[0][0] += A₀×B₀ + A₁×B₁ + A₂×B₂ + ...   │
└───────────────────────────────────────────────┘

각 TILE×TILE 블록:
  1. A의 타일 하나를 공유 메모리에 로드  (전역에서 TILE² 읽기)
  2. B의 타일 하나를 공유 메모리에 로드  (전역에서 TILE² 읽기)
  3. 부분 내적 계산                      (레지스터에서 TILE² 곱셈+덧셈)
  4. K 차원의 모든 타일에 대해 반복
```

블록당 전역 메모리 읽기: `2 × TILE² × (N/TILE)` = `2 × N × TILE`

단순 방식 대비: `N²` 읽기 (각 스레드가 N개 원소 읽기, TILE² 스레드)

**산술 강도 개선**: `N / (2 × TILE)` × 향상

---

## 3. 타일된 행렬 곱셈 구현

```c
// matmul_tiled.cu
#include <cuda_runtime.h>
#define TILE 16

__global__ void matmul_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;  // 레지스터에 누산기

    // K 차원을 따라 타일을 반복
    for (int t = 0; t < N / TILE; t++) {

        // 1단계: A와 B의 타일을 공유 메모리에 로드
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];

        // 2단계: 동기화 — 모든 스레드가 로드를 완료해야 연산 시작
        __syncthreads();

        // 3단계: 이 타일에 대한 부분 내적 계산
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // 4단계: 동기화 — 다음 타일 로드 전에 연산 완료
        __syncthreads();
    }

    // 최종 결과 쓰기 (범위 내에 있을 때만)
    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

### 핵심: 두 개의 `__syncthreads()` 배리어

```
배리어 1 없이 (로드 후):
  스레드 0이 스레드 15가 원소를 로드하기 전에 연산 시작할 수 있음.
  → 잘못된 결과.

배리어 2 없이 (다음 타일 로드 전):
  스레드 0이 다음 타일의 새 값으로 As[0][0]을 덮어쓸 수 있음
  스레드 15가 현재 타일의 As[0][0]을 아직 읽고 있는 동안.
  → 잘못된 결과 (데이터 경쟁).
```

`__syncthreads()`는 블록의 모든 스레드가 배리어에 도달해야 어떤 스레드도 진행할 수 있음을 보장합니다. 블록 간에는 효과가 없습니다.

---

## 4. 비정방/비타일배수 크기 처리

위의 타일된 커널은 N이 TILE의 배수라고 가정합니다. 견고한 버전:

```c
__global__ void matmul_tiled_safe(const float *A, const float *B, float *C,
                                   int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // 범위 밖 로드를 0으로 처리
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ?
                                        A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ?
                                        B[bRow * N + col]  : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

---

## 5. 공유 메모리 뱅크 충돌

공유 메모리는 **32개 뱅크** (Ampere)로 구성되며, 각 뱅크는 4바이트 폭입니다. 뱅크는 사이클당 하나의 요청을 처리합니다. **뱅크 충돌**은 warp의 여러 스레드가 **같은 뱅크**의 서로 다른 주소에 접근할 때 발생합니다 — 접근이 직렬화됩니다.

```
뱅크 레이아웃 (32 뱅크, 4바이트 워드):
  주소 0  → 뱅크 0     주소 32 → 뱅크 0
  주소 4  → 뱅크 1     주소 36 → 뱅크 1
  주소 8  → 뱅크 2     ...
  ...
  주소 124 → 뱅크 31   주소 156 → 뱅크 31

주소 a의 뱅크 번호: (a / 4) % 32
```

### 예: 전치에서 2방향 뱅크 충돌

warp에서 `tile[threadIdx.x][threadIdx.y]` 읽기:
- 스레드 0이 `tile[0][0]` → 뱅크 0 읽기
- 스레드 1이 `tile[1][0]` → 뱅크 0 (충돌! 오프셋 = 16 원소 = 64바이트, 뱅크 = 0)
- ...
- 스레드 31이 `tile[31][0]` → 뱅크 0

32개 스레드 모두 뱅크 0을 공격 → 32방향 뱅크 충돌 → 32배 느림.

### 해결: 패딩

```c
__shared__ float tile[TILE][TILE + 1];  // 패딩 열 1개 추가
```

패딩 적용 시:
- `tile[0][0]` → 뱅크 0, `tile[1][0]` → 뱅크 (16+1)%32 = 17, `tile[2][0]` → 뱅크 (34)%32 = 2

이제 각 열 접근이 서로 다른 뱅크를 사용 → 충돌 없음.

```c
// 뱅크 충돌 확인 (디버그 전용)
// 실행: ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
```

---

## 6. 성능 비교

```
행렬 곱셈 1024×1024 (FP32), RTX 3090에서:

  단순 커널:          ~3.2 ms  ~  670 GFLOPS  (최대 FP32의 9.6%)
  타일 TILE=16:       ~0.54 ms ~ 3,970 GFLOPS  (최대의 56%)
  타일 TILE=32:       ~0.41 ms ~ 5,230 GFLOPS  (최대의 75%)
  cuBLAS:             ~0.37 ms ~ 5,800 GFLOPS  (최대의 83%)
```

타일된 커널은 간단한 공유 메모리 최적화로 ~4배 속도 향상을 달성합니다.

---

## 7. Nsight Compute로 프로파일링

```bash
# 공유 메모리 메트릭 프로파일링
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second \
    ./matmul_tiled

# 확인할 핵심 메트릭:
# - bank_conflicts = 0 (또는 매우 낮음) ✓
# - warps_active > 50% (좋은 점유율) ✓
# - 전역 로드 바이트/s가 최대 대역폭에 근접 ✓
```

---

## 8. 레지스터 타일링 (L32 미리 보기)

타일링을 더 발전시킬 수 있습니다: 각 스레드가 레지스터를 사용하여 **출력 원소의 타일**을 계산합니다:

```c
#define RX 4  // 각 스레드가 RX×RY 출력 원소 계산
#define RY 4

__global__ void matmul_register_tiled(...) {
    float C_reg[RY][RX] = {};  // 레지스터 누산기 — RY×RX 출력 값

    // ... 공유 메모리에 타일 로드 후:
    for (int k = 0; k < TILE; k++) {
        float a_reg[RY];   // 레지스터 파일 — A 타일의 열 하나
        float b_reg[RX];   // 레지스터 파일 — B 타일의 행 하나
        // 공유 메모리에서 a_reg, b_reg 로드 (빠름)
        for (int ry = 0; ry < RY; ry++)
            for (int rx = 0; rx < RX; rx++)
                C_reg[ry][rx] += a_reg[ry] * b_reg[rx];
    }
    // C_reg를 전역 메모리에 다시 씀
}
```

이는 레지스터의 데이터를 재사용하여 (0사이클) 산술 강도를 더욱 높입니다. L32에서 이를 완전한 고성능 GEMM으로 발전시킵니다.

---

## 핵심 요약

- **공유 메모리**는 소프트웨어로 관리되는 캐시 역할 — 한 번 로드하고 블록 내에서 여러 번 사용
- 타일된 행렬 곱셈은 전역 메모리 읽기를 O(N³)에서 O(N²√N)으로 감소 — √N 개선
- **타일 반복당 두 개의 `__syncthreads()`** 필수: 로드 후 (연산 전)와 다음 로드 전
- **뱅크 충돌**은 warp 접근을 직렬화 — 열 충돌을 없애려면 공유 배열을 1원소 패딩
- `ncu`로 프로파일링하여 확인: 뱅크 충돌 없음, 높은 점유율, 최대 메모리 처리량에 근접

---

**다음**: [06. Warp 실행과 분기 발산](./06_Warp_Execution_and_Divergence.md) — 분기 발산이 warp를 직렬화하는 방법, warp 레벨 셔플 리덕션 구현, `__shfl_sync`를 통한 스레드 간 통신.
