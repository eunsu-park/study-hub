# 30. 혼합 정밀도와 Tensor Core

**이전**: [cuBLAS and cuSPARSE](./29_cuBLAS_and_cuSPARSE.md) | **다음**: [Cooperative Groups](./31_Cooperative_Groups.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. FP32, FP16, BF16, FP8의 수치 특성과 트레이드오프 설명하기
2. WMMA API (`nvcuda::wmma`)를 사용하여 Tensor Core를 직접 사용하는 CUDA kernel 작성하기
3. 16×16×16 WMMA 행렬 곱셈-누적 연산 구현하기
4. 훈련 중 FP16 그래디언트 언더플로를 방지하기 위한 손실 스케일링 적용하기
5. Tensor Core FLOPS vs CUDA 코어 FLOPS 측정 및 각각이 지배적인 경우 이해하기

---

## 1. 부동 소수점 형식

```
형식      비트   지수부   가수부   동적 범위          참고
----------------------------------------------------------------------
FP64       64      11        52        ±10^±308           CPU 기본값
FP32       32       8        23        ±10^±38            GPU 기본값
FP16       16       5        10        ±65504             IEEE 754 half
BF16       16       8         7        ±10^±38            "Brain Float" (Google TPU)
FP8 E4M3   8        4         3        ±448               CUDA 12+ (Hopper)
FP8 E5M2   8        5         2        ±57344             더 넓은 범위, 그래디언트용
TF32       19      (FP32의 부분집합)  ±10^±38            A100 내부 Tensor Core 형식

핵심 트레이드오프:
  FP16: 좋은 정밀도, 제한된 범위 → 그래디언트에 손실 스케일링 필요
  BF16: FP32와 동일한 범위, 낮은 정밀도 → FP32 범위의 드롭인 대체품
  FP8:  Hopper에서 FP16의 2배 처리량; 신중한 양자화 필요
```

---

## 2. CUDA에서의 FP16 데이터 타입

```c
#include <cuda_fp16.h>

// half: 16비트 float 타입
__global__ void half_demo() {
    half a = __float2half(3.14f);    // FP32 → FP16
    half b = __float2half(2.71f);
    half c = __hadd(a, b);           // FP16 덧셈
    half d = __hmul(a, b);           // FP16 곱셈
    float f = __half2float(c);       // FP16 → FP32

    // half2: 32비트에 패킹된 두 개의 FP16 값 (SIMD 2× 처리량)
    half2 v = __float22half2_rn(make_float2(1.f, 2.f));
    half2 w = __float22half2_rn(make_float2(3.f, 4.f));
    half2 r = __hadd2(v, w);        // 패킹된 2× FP16 덧셈
}

// BF16은 cuda_bf16.h 필요 (CUDA 11.0+)
#include <cuda_bf16.h>
__global__ void bf16_demo() {
    __nv_bfloat16 a = __float2bfloat16(3.14f);
    __nv_bfloat16 b = __float2bfloat16(2.71f);
    __nv_bfloat16 c = __hadd(a, b);  // FP16과 동일한 내장 함수
    float f = __bfloat162float(c);
}
```

---

## 3. Tensor Core 개요

Tensor Core는 특수화된 행렬 곱셈-누적 (MMA) 장치입니다:

```
A100 Tensor Core 성능 (SM당, 클럭당):
  FP16 TC:   256 FLOPs
  BF16 TC:   256 FLOPs
  TF32 TC:   128 FLOPs
  FP64 TC:    64 FLOPs
  INT8 TC:   512 OPs

vs CUDA 코어:
  FP32 CUDA:   2 FLOPs (1 FMA)
  FP16 CUDA:   2 FLOPs

따라서 Tensor Core는 장치당 클럭당 128-256배 더 효율적입니다.

WMMA fragment 크기 (Tensor Core 하드웨어와 일치해야 함):
  16×16×16  (FP16/BF16 FP32 또는 FP16으로 누적)
  8×16×16   (일부 GPU에 대한 대안)
  32×8×16   (대안)

정렬 요구 사항: 병합된 로드를 위해 행렬이 16-요소 정렬되어야 함.
```

---

## 4. WMMA API

WMMA (Warp Matrix Multiply Accumulate) API는 warp 수준에서 Tensor Core를 노출합니다. 하나의 warp (32개 thread)가 협력하여 16×16 행렬 fragment를 보유합니다:

```c
#include <mma.h>
using namespace nvcuda;

// 16×16×16 FP16 → FP32 WMMA를 위한 fragment 타입
// 각 warp가 하나의 fragment를 협력하여 보유 (32개 thread에 분산)
using frag_a   = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
using frag_b   = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
using frag_acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// WMMA GEMM: C[16×16] += A[16×16] * B[16×16]
// 출력 tile당 하나의 warp
__global__ void wmma_gemm_16x16(
    const half *A, const half *B, float *C,
    int M, int N, int K)
{
    // 출력에서의 warp 위치
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * 16;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;

    if (warp_row >= M || warp_col >= N) return;

    frag_a   a_frag;
    frag_b   b_frag;
    frag_acc c_frag;

    // 누적기를 0으로 초기화
    wmma::fill_fragment(c_frag, 0.f);

    // K에 대해 16-넓이 tile로 반복
    for (int k = 0; k < K; k += 16) {
        // A tile 로드: 행 warp_row, 열 k에 대한 포인터
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);

        // B tile 로드: 행 k, 열 warp_col에 대한 포인터 (B는 열 주요)
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);

        // 행렬 곱셈-누적: c_frag += a_frag * b_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 결과 저장
    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N,
                             wmma::mem_row_major);
}
```

---

## 5. Shared Memory를 사용한 WMMA (프로덕션 패턴)

원시 전역 메모리 WMMA는 대역폭 제한을 받습니다. 실제 구현은 shared memory로 tile을 처리합니다:

```c
// 128×128 block tile, 16×16의 8×8 warp tile로 처리
// cuBLAS가 내부적으로 사용하는 구조입니다

#define BM 128   // block M tile
#define BN 128   // block N tile
#define BK 16    // block K tile (Tensor Core 내부 차원)

__global__ void wmma_tiled(const half *A, const half *B, float *C,
                            int M, int N, int K) {
    __shared__ half sA[BM][BK];
    __shared__ half sB[BK][BN];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    // block 내의 warp 격자: 4×4 배열 (block당 16개 warp)
    int warp_row = warp_id / 4;  // 0..3 (각각 32행 처리)
    int warp_col = warp_id % 4;  // 0..3 (각각 32열 처리)

    frag_acc c_frag[2][2];  // warp당 2×2 16×16 누적기
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(c_frag[i][j], 0.f);

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    for (int k = 0; k < K; k += BK) {
        // A[BM×BK]와 B[BK×BN]을 shared memory에 협력하여 로드
        // (각 thread가 여러 요소를 로드; 명확성을 위해 세부 사항 생략)
        load_tile_to_shared(A, sA, block_row, k, M, K);
        load_tile_to_shared_B(B, sB, k, block_col, K, N);
        __syncthreads();

        // 각 warp가 2×2 WMMA tile 계산
        for (int wi = 0; wi < 2; wi++) {
            for (int wj = 0; wj < 2; wj++) {
                frag_a a_frag; frag_b b_frag;
                int row_off = (warp_row * 2 + wi) * 16;
                int col_off = (warp_col * 2 + wj) * 16;
                wmma::load_matrix_sync(a_frag, &sA[row_off][0], BK);
                wmma::load_matrix_sync(b_frag, &sB[0][col_off], BN);
                wmma::mma_sync(c_frag[wi][wj], a_frag, b_frag, c_frag[wi][wj]);
            }
        }
        __syncthreads();
    }

    // 누적기 저장
    for (int wi = 0; wi < 2; wi++) {
        for (int wj = 0; wj < 2; wj++) {
            int row = block_row + (warp_row*2+wi)*16;
            int col = block_col + (warp_col*2+wj)*16;
            if (row < M && col < N)
                wmma::store_matrix_sync(C + row*N + col, c_frag[wi][wj], N,
                                        wmma::mem_row_major);
        }
    }
}
```

---

## 6. FP16 훈련을 위한 손실 스케일링

FP16 그래디언트는 ~6×10^-5 (FP16 최솟값 정규) 미만의 값에서 0으로 언더플로될 수 있습니다. 손실 스케일링은 역전파 전에 손실에 큰 상수를 곱한 후 그래디언트를 나눕니다:

```c
// 동적 손실 스케일링 (PyTorch AMP 방식)
float loss_scale = 65536.f;   // 초기 스케일 팩터
int growth_interval = 2000;   // 스케일 증가 간격 (단계)
int n_skipped = 0;

for (int step = 0; step < total_steps; step++) {
    // 순전파 (혼합 정밀도에서 FP32 누적)
    // 스케일된 손실로 역전파:
    //   scaled_loss = loss * loss_scale
    //   gradients  *= loss_scale

    // 그래디언트에서 inf/nan 확인
    bool has_inf = check_gradients_for_inf(d_grads, param_count);

    if (has_inf) {
        loss_scale /= 2.f;   // 스케일 감소
        n_skipped++;
        printf("단계 %d: 건너뜀 (오버플로), 스케일 → %.0f\n", step, loss_scale);
        continue;  // 매개변수 업데이트 건너뜀
    }

    // 그래디언트 언스케일
    scale_gradients<<<GRID, BLOCK>>>(d_grads, 1.f / loss_scale, param_count);

    // 옵티마이저 단계...

    // 주기적으로 스케일 증가
    if ((step + 1) % growth_interval == 0 && n_skipped == 0) {
        loss_scale = fminf(loss_scale * 2.f, 65536.f);
    }
    n_skipped = 0;
}

// Inf/NaN 확인 kernel
__global__ void check_inf_kernel(const float *grad, int *flag, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && !isfinite(grad[i]))
        atomicExch(flag, 1);   // 오버플로 신호
}
```

---

## 7. Tensor Core FLOPS 측정

```c
// nvcc -arch=sm_80으로 프로파일링, 그 다음 ncu로 비교:
// ncu --metrics sm__ops_warps_eligible.avg,
//               sm__inst_executed_pipe_tensor.avg
//               ./my_gemm

// 간단한 호스트 측 측정
void measure_flops(int M, int N, int K, int iters) {
    // d_A(M×K), d_B(K×N), d_C(M×N)을 half/float로 할당...

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; i++)
        wmma_gemm_16x16<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = 2.0 * M * N * K * iters;  // FMA당 2 FLOPs
    double tflops = flops / (ms * 1e9);
    printf("WMMA: %.1f TFLOPS (%d회 반복에 %.2f ms)\n", tflops, iters, ms);

    // 비교: CUDA 코어 FP32 GEMM
    // A100: 312 TFLOPS (TC FP16), 19.5 TFLOPS (CUDA FP32)
    // 비율 ≈ FP16 Tensor Core vs FP32 CUDA 코어에서 16배
}
```

---

## 핵심 요약

- **FP16**은 5비트 지수 (범위 ±65504)와 10비트 가수를 가집니다; **BF16**은 8비트 지수 (FP32와 동일한 범위)와 7비트 가수를 가집니다 — BF16이 훈련 안정성에 선호됩니다
- **WMMA API**: 하나의 warp (32개 thread)가 협력하여 16×16 fragment를 보유합니다; `load_matrix_sync` → `mma_sync` → `store_matrix_sync`가 전체 패턴입니다
- **Fragment 레이아웃**은 32개 warp thread에 걸쳐 하드웨어 정의 및 불투명합니다; 특정 thread-요소 매핑에 의존하지 마세요
- **Tile 기반 WMMA**는 `mma_sync` 호출 전에 A와 B 서브 tile을 shared memory에 로드합니다; 이는 Tensor Core 피크 처리량에 근접하는 데 중요합니다
- **손실 스케일링**은 손실에 큰 상수 (예: 2^16)를 곱하여 작은 그래디언트를 FP16 최솟값 위로 이동시킵니다; 동적 손실 스케일링이 팩터를 자동으로 조정합니다
- A100에서: FP16 Tensor Core는 ~312 TFLOPS vs FP32 CUDA 코어의 ~19.5 TFLOPS — 16배 이론적 차이, 실제에서는 ~8-12배 측정됨

---

**다음**: [31. Cooperative Groups](./31_Cooperative_Groups.md) — CUDA Cooperative Groups를 사용하여 하드코딩된 `__syncthreads()` 없이 warp, block, grid 범위에서 작동하는 유연한 thread 조율 코드를 작성합니다.
