# 29. cuBLAS와 cuSPARSE

**이전**: [Thrust and CUB](./28_Thrust_and_CUB.md) | **다음**: [Mixed Precision and Tensor Cores](./30_Mixed_Precision_and_Tensor_Cores.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. cuBLAS handle 초기화 및 밀집 행렬 곱셈을 위한 `cublasSgemm` 호출하기
2. cuBLAS의 열 주요(column-major) 규칙 이해 및 행 주요(row-major) C 배열 올바르게 매핑하기
3. 많은 소형 행렬을 위해 `cublasGemmBatchedEx`로 배치 GEMM 수행하기
4. `CUDA_R_16F`를 통한 Tensor Core 가속으로 `cublasGemmEx` 사용하기
5. 희소 행렬을 CSR 형식으로 포맷하고 cuSPARSE로 희소 행렬-벡터 곱셈 (SpMV) 수행하기

---

## 1. cuBLAS Handle 및 설정

모든 cuBLAS 함수는 CUDA 컨텍스트, stream, 작업 공간을 캡슐화하는 `cublasHandle_t`가 필요합니다:

```c
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// 비기본 stream과 연결 (선택적)
cudaStream_t stream;
cudaStreamCreate(&stream);
cublasSetStream(handle, stream);

// 완료 시 항상 해제
// cublasDestroy(handle);
```

**오류 검사 매크로:**

```c
#define CUBLAS_CHECK(call) do {                                 \
    cublasStatus_t status = call;                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                      \
        fprintf(stderr, "cuBLAS 오류 %d at %s:%d\n",          \
                status, __FILE__, __LINE__);                    \
        exit(1);                                                \
    }                                                           \
} while(0)
```

---

## 2. cuBLAS 열 주요 규칙

cuBLAS는 **Fortran (열 주요)** 규칙을 따릅니다. 행 주요 C 배열에는 **인수를 전치하거나 M/N을 교환**해야 합니다:

```
cuBLAS 계산: C = α·op(A)·op(B) + β·C

행 주요 행렬 A(M×K), B(K×N), C(M×N)의 경우:
  열 주요 B^T(N×K), A^T(K×M), C^T(N×M)로 전달
  → cublasSgemm이 C^T = α·B^T·A^T + β·C^T를 계산
  → 이는 (C = α·A·B + β·C)^T와 같음 ← 올바름!

트릭: 인수 A↔B 교환, M↔N 교환:
  cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,           // ← M과 N이 교환됨
    &alpha,
    d_B, N,            // ← B가 먼저, leading dim = N
    d_A, K,            // ← A가 두 번째, leading dim = K
    &beta,
    d_C, N);           // ← 출력 leading dim = N
```

```c
// 완전한 예시: C = A * B, A는 M×K, B는 K×N, C는 M×N (모두 행 주요)
void sgemm_rowmajor(cublasHandle_t handle,
                    const float *d_A, const float *d_B, float *d_C,
                    int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;

    // 행 주요 트릭: A↔B 교환 및 M↔N 교환
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N,      // op(B)의 행 = N
        M,      // op(A)의 열 = M
        K,      // 내부 차원
        &alpha,
        d_B, N, // B: leading dim = N (행 주요에서 각 행은 N개 요소)
        d_A, K, // A: leading dim = K
        &beta,
        d_C, N  // C: leading dim = N
    ));
}
```

---

## 3. 성능: cuBLAS vs 나이브 GEMM

```
행렬 크기 4096×4096, FP32, A100 GPU:

Kernel              시간     TFLOPS
-----------------------------------
우리의 L32 kernel v2  1.8 ms    75.6
우리의 L32 kernel v3  1.1 ms   124.8   (register tiling)
cublasSgemm          0.65 ms  211.5   (내부적으로 Tensor Core 사용)
이론적 피크           0.42 ms  312     (A100 FP32 dense)

FP16의 경우 (Tensor Core):
cublasHgemm          0.17 ms  ~600 TFLOPS (A100은 312 TF FP16 TC 보유)
```

---

## 4. 배치 GEMM

배치 GEMM은 한 번의 호출로 많은 독립적인 행렬 곱셈을 실행합니다 — 미니 배치에서 동작하는 신경망 레이어에 필수적입니다:

```c
// 배치 GEMM: i = 0..batch_size-1에 대해 C_i = A_i * B_i 계산
// 모든 행렬은 동일한 모양: A(M×K), B(K×N), C(M×N)
void batched_gemm(cublasHandle_t handle,
                  int M, int N, int K, int batch_size) {
    float alpha = 1.f, beta = 0.f;

    // 방법 1: cublasGemmBatchedEx — 포인터 배열
    // 각 d_Aarray[i]는 다른 M×K 행렬을 가리킴

    float **d_Aarray, **d_Barray, **d_Carray;
    cudaMalloc(&d_Aarray, batch_size * sizeof(float*));
    cudaMalloc(&d_Barray, batch_size * sizeof(float*));
    cudaMalloc(&d_Carray, batch_size * sizeof(float*));

    // 호스트에서 포인터 배열을 채우고 device로 복사
    float **h_Aarray = (float**)malloc(batch_size * sizeof(float*));
    // ... 각 행렬을 할당하고 h_Aarray[i]를 채움 ...
    cudaMemcpy(d_Aarray, h_Aarray, batch_size*sizeof(float*), cudaMemcpyHostToDevice);

    CUBLAS_CHECK(cublasGemmBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (const void**)d_Barray, CUDA_R_32F, N,
        (const void**)d_Aarray, CUDA_R_32F, K,
        &beta,
        (void**)d_Carray,       CUDA_R_32F, N,
        batch_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

    // 방법 2: cublasGemmStridedBatchedEx — 연속 스트라이드 (더 효율적)
    // A[i] = base_A + i * stride_A 등을 가정
    long long stride_A = M * K, stride_B = K * N, stride_C = M * N;
    float *d_A, *d_B, *d_C;
    // ... batch_size * M*K float을 d_A 등에 할당 ...

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_32F, N, stride_B,
        d_A, CUDA_R_32F, K, stride_A,
        &beta,
        d_C, CUDA_R_32F, N, stride_C,
        batch_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}
```

---

## 5. Tensor Core GEMM (FP16)

Tensor Core는 16×16×16 FP16 행렬 곱셈-누적을 단일 명령어로 계산합니다. `CUDA_R_16F`와 `CUBLAS_COMPUTE_32F_FAST_16F`로 활성화합니다:

```c
#include <cuda_fp16.h>

void gemm_tensor_cores(cublasHandle_t handle,
                       const half *d_A, const half *d_B, float *d_C,
                       int M, int N, int K) {
    float alpha = 1.f, beta = 0.f;

    // cublasGemmEx: 명시적 타입 지정
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16F, N,    // B는 FP16
        d_A, CUDA_R_16F, K,    // A는 FP16
        &beta,
        d_C, CUDA_R_32F, N,    // C는 FP32로 누적
        CUBLAS_COMPUTE_32F_FAST_16F,   // Tensor Core 사용
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// 호출 전에 FP32 배열을 FP16으로 변환
__global__ void f32_to_f16(const float *in, half *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __float2half(in[i]);
}
```

---

## 6. cuSPARSE: CSR 형식

희소 행렬은 압축 희소 행 (CSR) 형식으로 압축하여 저장합니다:

```
밀집 4×4 행렬:
  0  3  0  0
  2  0  0  5
  0  1  4  0
  0  0  0  6

CSR 표현:
  values   = [3, 2, 5, 1, 4, 6]          (행 주요 순서의 non-zero)
  col_idx  = [1, 0, 3, 1, 2, 3]          (각 non-zero의 열 인덱스)
  row_ptr  = [0, 1, 3, 5, 6]             (values에서 각 행의 시작; 길이 = nrows+1)

nnz = 6 (non-zero 수)
압축 비율: 6/16 = 37.5%의 밀집 저장소
```

---

## 7. cuSPARSE SpMV

CSR 형식을 사용한 희소 행렬-벡터 곱셈 y = A·x:

```c
#include <cusparse.h>

void spmv_csr(
    int nrows, int ncols, int nnz,
    const int *d_row_ptr, const int *d_col_idx, const float *d_values,
    const float *d_x, float *d_y)
{
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // 행렬 디스크립터 생성
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(
        &matA,
        nrows, ncols, nnz,
        (void*)d_row_ptr, (void*)d_col_idx, (void*)d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // 벡터 디스크립터 생성
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, ncols, (void*)d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, nrows, (void*)d_y, CUDA_R_32F);

    float alpha = 1.f, beta = 0.f;

    // 버퍼 크기 조회
    void   *d_buf = nullptr;
    size_t  buf_bytes = 0;
    cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_bytes);
    cudaMalloc(&d_buf, buf_bytes);

    // SpMV 실행
    cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);

    // 정리
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cudaFree(d_buf);
    cusparseDestroy(handle);
}
```

---

## 8. 성능: 밀집 vs 희소

```
SpMV는 메모리 대역폭 바운드; 희소성의 이점은 nnz/N 비율에 따라 다릅니다.

행렬: N=100,000 행, K=100,000 열
케이스 A: 밀집 A (100억 요소) → 900 GB/s에서 GEMV: ~88ms
케이스 B: 희소 A (1% 밀도, 1억 nnz) → SpMV: ~2ms
  → 99% zeros일 때 44배 빠름

실제 딥러닝 (BERT attention, 50% 희소성):
  밀집 GEMM:       0.8 ms
  희소 (CSR):      1.2 ms  ← 보통 희소성에서는 종종 더 느림!
  이유: CSR은 메모리 접근 패턴이 나쁨; 속도 향상에는 >90% 희소성 필요

구조적 희소성 (2:4 형식 — 4요소 중 2개 non-zero):
  NVIDIA Ampere cuSPARSE 구조적:  정확히 50% 희소에서 밀집보다 ~1.5배 빠름
  특정 패턴 필요; Ampere 희소 Tensor Core에서 사용
```

---

## 9. cuSPARSE SpMM (희소 × 밀집 행렬)

```c
// y = A * B 여기서 A는 희소 CSR (nrows×k), B는 밀집 (k×ncols)
void spmm_csr(cusparseHandle_t handle,
              cusparseSpMatDescr_t matA,
              const float *d_B, float *d_C,
              int nrows, int ncols_B, int k) {
    cusparseDnMatDescr_t matB, matC;
    cusparseCreateDnMat(&matB, k,     ncols_B, ncols_B, (void*)d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, nrows, ncols_B, ncols_B, (void*)d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.f, beta = 0.f;
    void *d_buf = nullptr; size_t buf_bytes = 0;
    cusparseSpMM_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_bytes);
    cudaMalloc(&d_buf, buf_bytes);
    cusparseSpMM(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf);
    cusparseDestroyDnMat(matB); cusparseDestroyDnMat(matC);
    cudaFree(d_buf);
}
```

---

## 핵심 요약

- **cuBLAS 행 주요 트릭**: 열 주요 cuBLAS에 행 주요 C 배열을 사용하려면 `cublasSgemm` 호출에서 A↔B를 교환하고 M↔N을 교환하세요
- **Leading dimension**은 연속된 열 사이의 스트라이드 (열 주요)입니다: 행 주요 M×N 행렬의 경우 leading dimension은 N입니다
- **배치 GEMM**: `cublasGemmBatchedEx` (포인터 배열)와 `cublasGemmStridedBatchedEx` (고정 스트라이드)는 한 번의 호출로 독립적인 미니 배치 행렬 곱셈을 처리합니다
- **Tensor Core**: `CUDA_R_16F` 입력과 함께 `CUBLAS_COMPUTE_32F_FAST_16F`로 활성화; FP32 CUDA 코어의 2-4배 FLOPS를 제공할 수 있습니다
- **CSR 형식**: nnz 값을 해당 열 인덱스와 행 포인터 배열로 저장; <10% 밀도에서 메모리 효율적
- **SpMV 성능**: 희소는 높은 희소성 (>80-90%)에서만 밀집보다 빠릅니다; Ampere에서 구조적 2:4 희소성은 일관적으로 ~2배 속도 향상을 제공합니다

---

**다음**: [30. Mixed Precision and Tensor Cores](./30_Mixed_Precision_and_Tensor_Cores.md) — FP16, BF16, WMMA API를 활용하여 Tensor Core 행렬 연산을 직접 프로그래밍하는 CUDA kernel을 작성합니다.
