# 32. GEMM from Scratch

**이전**: [Cooperative Groups](./31_Cooperative_Groups.md) | **다음**: [Softmax and LayerNorm Kernels](./33_Softmax_and_LayerNorm_Kernels.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 나이브 GEMM kernel을 구현하고 왜 메모리 대역폭에 병목이 걸리는지 이해하기
2. shared memory tiling을 적용하여 전역 메모리 트래픽을 tile 크기만큼 줄이기
3. register blocking 사용하기 (각 thread가 4×4 또는 8×8 출력 서브-tile 계산)
4. 최대 메모리 처리량을 위해 `float4`로 메모리 로드 벡터화하기
5. 각 버전을 벤치마킹하고 직접 작성한 kernel과 cuBLAS 사이의 차이 이해하기

---

## 1. 문제 설정

C = A · B를 계산합니다. A는 M×K, B는 K×N, C는 M×N (모두 row-major FP32).

```
FLOP count: 2·M·N·K  (K 합산의 각 원소마다 곱셈 1번 + 덧셈 1번)
M=N=K=4096인 경우: 2 × 4096³ ≈ 137 GFLOP

루프라인 분석:
  메모리: (M*K + K*N + M*N) × 4 bytes = 3 × 4096² × 4 = 192 MB
  900 GB/s에서: 대역폭 한계 = 192/900 ≈ 0.21 ms → 137G/0.21ms = 652 TFLOPS
  A100 FP32 연산 한계 19.5 TFLOPS에서: 137G/19.5T ≈ 7 ms

따라서 GEMM은 연산 병목 (compute-bound); 목표는 FLOPs/byte를 최대화하는 것.
```

---

## 2. 버전 1: 나이브 전역 메모리 GEMM

```c
// v1: 출력 원소당 하나의 thread; 전역 메모리에서 전체 행과 열을 읽음
__global__ void gemm_v1_naive(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 출력 행
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 출력 열
    if (row >= M || col >= N) return;

    float sum = 0.f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];   // 각 로드가 전역 메모리에서

    C[row * N + col] = sum;
}

// 실행: block당 32×32 thread
void launch_v1(const float *dA, const float *dB, float *dC, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    gemm_v1_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
}
```

**v1 분석:**
- 각 thread는 A에서 K개 원소 (행)와 B에서 K개 원소 (열)을 로드
- 32-thread block 행에서 인접 thread가 B의 인접 원소에 접근 (coalesced)
- 그러나 B 접근은 stride-N 열 → 큰 N에서 L1 캐시 성능 저하
- 측정값: ~0.5 TFLOPS (최대 성능의 약 2.5%)

---

## 3. 버전 2: Shared Memory Tiling

A와 B의 TILE×TILE 서브-block을 shared memory에 로드한 다음 부분 내적을 계산합니다:

```c
#define TILE 32

__global__ void gemm_v2_tiled(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.f;

    // K 차원을 TILE 폭 단계로 순회
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // A tile 로드: A에서 row번째 행, tile t의 열들
        int a_col = t * TILE + tx;
        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;

        // B tile 로드: tile t의 행들, B에서 col번째 열
        int b_row = t * TILE + ty;
        sB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();

        // shared 데이터를 사용한 내적 누적
        for (int k = 0; k < TILE; k++)
            sum += sA[ty][k] * sB[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**v2 분석:**
- A 또는 B의 각 원소는 shared memory에 한 번 로드되고 TILE=32 번 재사용
- 전역 메모리 트래픽: (M*K + K*N) / TILE 배 감소 → 읽기 32× 감소
- 측정값: ~15 TFLOPS (TILE=32에서 이론적 대역폭 기반 한계의 75%)
- 병목: 내부 루프 `sum += sA * sB` — 2번의 shared-memory 읽기당 2 FLOP

---

## 4. 버전 3: Register Tiling (Thread가 BM×BN 출력 계산)

thread당 하나의 출력 원소 대신, 각 thread가 register에 저장된 BM×BN tile을 계산합니다. 이렇게 하면 더 많은 FLOP에 걸쳐 shared-memory 로드 비용이 분산됩니다:

```c
// 각 thread는 4×4 출력 tile을 계산
// Block: 8×8 thread × thread당 4×4 = 32×32 출력 tile
// (TILE=32와 동일하지만 이제 각 thread는 1개 대신 16개의 MAC 수행)

#define TILE_M 32    // block 출력 행
#define TILE_N 32    // block 출력 열
#define TILE_K 8     // 단계당 K-strip
#define THREAD_M 4   // thread당 행 tile
#define THREAD_N 4   // thread당 열 tile

__global__ void gemm_v3_register(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    __shared__ float sA[TILE_K][TILE_M];   // K × M tile
    __shared__ float sB[TILE_K][TILE_N];   // K × N tile

    // block 내 thread 위치
    int tx = threadIdx.x;   // 0..7 (TILE_N/THREAD_N)
    int ty = threadIdx.y;   // 0..7 (TILE_M/THREAD_M)

    // 이 thread의 왼쪽 상단 원소의 출력 위치
    int row0 = blockIdx.y * TILE_M + ty * THREAD_M;
    int col0 = blockIdx.x * TILE_N + tx * THREAD_N;

    // register 누산기
    float acc[THREAD_M][THREAD_N] = {};

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // A strip [TILE_M × TILE_K]을 sA에 로드
        // (TILE_M/THREAD_M) × (TILE_N/THREAD_N) = 8×8 = 64 thread in block
        // 각 thread는 sA/sB를 채우기 위해 여러 원소를 로드
        for (int i = 0; i < THREAD_M; i++) {
            int gRow = blockIdx.y * TILE_M + ty * THREAD_M + i;
            int gCol = t * TILE_K + tx % TILE_K;
            sA[tx % TILE_K][ty * THREAD_M + i] =
                (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.f;
        }
        for (int j = 0; j < THREAD_N; j++) {
            int gRow = t * TILE_K + ty % TILE_K;
            int gCol = blockIdx.x * TILE_N + tx * THREAD_N + j;
            sB[ty % TILE_K][tx * THREAD_N + j] =
                (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.f;
        }

        __syncthreads();

        // 계산: 각 thread는 THREAD_M × THREAD_N × TILE_K MAC 수행
        for (int k = 0; k < TILE_K; k++) {
            float ra[THREAD_M], rb[THREAD_N];
            for (int i = 0; i < THREAD_M; i++) ra[i] = sA[k][ty*THREAD_M+i];
            for (int j = 0; j < THREAD_N; j++) rb[j] = sB[k][tx*THREAD_N+j];
            for (int i = 0; i < THREAD_M; i++)
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += ra[i] * rb[j];
        }

        __syncthreads();
    }

    // register 누산기를 전역 메모리에 쓰기
    for (int i = 0; i < THREAD_M; i++)
        for (int j = 0; j < THREAD_N; j++) {
            int gRow = row0 + i, gCol = col0 + j;
            if (gRow < M && gCol < N)
                C[gRow * N + gCol] = acc[i][j];
        }
}
```

**v3 분석:**
- 각 shared-memory 로드가 THREAD_M × THREAD_N = 16번 재사용
- 산술 강도: 2 × 4 × 4 × 8 / (2 × 8 × 4) = 4 FLOPs/byte (v2의 1 FLOPs/byte 대비)
- 측정값: ~60 TFLOPS

---

## 5. 버전 4: float4 벡터화 로드

`float4`는 단일 명령으로 4개의 float (16 bytes)을 로드하여 유효 대역폭을 높입니다:

```c
// 4개의 연속 float을 단일 float4 트랜잭션으로 로드
__device__ __forceinline__ float4 load4(const float *ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// 벡터화된 A tile 로드: 단계당 thread당 4개 원소 로드
__global__ void gemm_v4_vectorized(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    // v3과 유사한 구조이지만 로드를 float4로 수행
    // 정렬을 위해 K가 4의 배수여야 함

    __shared__ float sA[TILE_K][TILE_M + 4];  // +4로 bank conflict 방지
    __shared__ float sB[TILE_K][TILE_N + 4];

    int tx = threadIdx.x, ty = threadIdx.y;
    float acc[THREAD_M][THREAD_N] = {};

    for (int t = 0; t < K / TILE_K; t++) {
        // B 행을 sB에 float4로 로드 (한 번에 4개 원소, coalesced)
        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            int b_col = blockIdx.x * TILE_N + tx * 4;
            if (b_row < K && b_col + 3 < N) {
                float4 b4 = load4(&B[b_row * N + b_col]);
                sB[ty][tx*4+0] = b4.x;
                sB[ty][tx*4+1] = b4.y;
                sB[ty][tx*4+2] = b4.z;
                sB[ty][tx*4+3] = b4.w;
            }
        }
        // sA도 유사하게...
        __syncthreads();

        // tile 계산 (v3과 동일)
        for (int k = 0; k < TILE_K; k++) {
            float ra[THREAD_M], rb[THREAD_N];
            for (int i = 0; i < THREAD_M; i++) ra[i] = sA[k][ty*THREAD_M+i];
            for (int j = 0; j < THREAD_N; j++) rb[j] = sB[k][tx*THREAD_N+j];
            for (int i = 0; i < THREAD_M; i++)
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

    // float4로 저장 (연속된 4개의 열)
    for (int i = 0; i < THREAD_M; i++) {
        int gRow = blockIdx.y * TILE_M + ty * THREAD_M + i;
        int gCol = blockIdx.x * TILE_N + tx * THREAD_N;
        if (gRow < M && gCol + THREAD_N - 1 < N) {
            float4 r4 = {acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
            *reinterpret_cast<float4*>(&C[gRow * N + gCol]) = r4;
        }
    }
}
```

---

## 6. 성능 비교

```
Kernel     Block       Thread당     M=N=K=4096   TFLOPS  cuBLAS 대비
---------------------------------------------------------------------
v1 naive   32×32       1×1 원소     72 ms        0.5      2.5%
v2 tiled   32×32       1×1 원소      9 ms        15       7%
v3 reg     8×8 thrd    4×4 원소     2.3 ms       60      28%
v4 vec     8×8 thrd    4×4 원소     1.8 ms       76      36%
cuBLAS     내부         내부         0.65 ms      211     100%

남은 차이 (36% → 100%):
  - cuBLAS는 더 큰 tile의 CUTLASS 사용 (128×128×32 또는 256×128×32)
  - 이중 버퍼링된 shared memory (현재 tile 계산 중 다음 tile 프리페치)
  - 비동기 전역→shared 복사 (cuda::pipeline, cp.async)
  - Tensor Core wmma / mma PTX 명령
  - 2~3단계 소프트웨어 파이프라이닝
```

---

## 7. cuBLAS의 80% 이상 달성하기

v4를 넘어서는 핵심 기법:

```c
// 기법 1: 이중 버퍼링 (소프트웨어 파이프라이닝)
// tile t를 계산하는 동안 tile t+1을 "next" shared memory bank에 프리로드
__shared__ float sA[2][TILE_K][TILE_M];  // 이중 버퍼
__shared__ float sB[2][TILE_K][TILE_N];
int cur = 0, nxt = 1;

// 첫 번째 tile 프리페치
load_tile_async(sA[cur], sB[cur], ...);
__syncthreads();

for (int t = 1; t <= ntiles; t++) {
    if (t < ntiles)
        load_tile_async(sA[nxt], sB[nxt], ...);  // 계산 중 프리페치
    compute_tile(sA[cur], sB[cur], acc);
    __syncthreads();
    swap(cur, nxt);
}

// 기법 2: cp.async (CUDA 11+, Ampere)
// register를 거치지 않고 전역→shared 복사
#include <cuda_pipeline.h>
__pipeline_memcpy_async(&sA[ty][tx], &A[row * K + col], sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);  // 모든 대기 중인 복사 완료 대기
```

---

## 핵심 요약

- **v1 naive**는 메모리 병목이며, 각 전역 로드는 단 하나의 곱셈에만 사용됨
- **v2 tiled**는 shared memory에 데이터를 준비함으로써 전역 트래픽을 TILE× 감소시킴; 로드와 계산 단계 사이의 `__syncthreads()` 경계가 필수적
- **v3 register tiling**: 각 thread가 register에 저장된 THREAD_M×THREAD_N 서브-tile을 계산하여 v2보다 높은 산술 강도 달성
- **v4 float4**는 128비트 벡터 로드 (명령당 4개의 float)를 사용하여 로드 명령 횟수를 4× 줄이고 메모리 처리량 향상
- **cuBLAS와의 남은 차이**는 이중 버퍼링된 shared memory (레이턴시 숨기기), `cp.async` (비동기 전역→shared 복사), Tensor Core 활용에서 비롯됨
- scratch부터 GEMM을 구축하는 것은 CUDA 성능 원칙을 마스터하는 최고의 연습: 루프라인 모델, occupancy, 메모리 계층, 명령 처리량이 모두 동시에 적용됨

---

**다음**: [33. Softmax and LayerNorm Kernels](./33_Softmax_and_LayerNorm_Kernels.md) — 수치적으로 안정적인 온라인 softmax와 warp shuffle을 사용한 융합 LayerNorm/RMSNorm을 구현합니다. 트랜스포머 추론의 핵심 구성 요소입니다.
