# 19. 희소 행렬 연산 (Sparse Matrix Operations)

**이전**: [Histogram and Binning](./18_Histogram_and_Binning.md) | **다음**: [N-Body Simulation](./20_N_Body_Simulation.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. COO, CSR, CSC 형식으로 희소 행렬을 표현하고 변환하기
2. 나이브 CSR SpMV kernel 구현하기 (행당 하나의 thread)
3. 부하 불균형을 해소하는 warp-per-row SpMV kernel 구현하기
4. 프로덕션 희소 행렬-벡터 곱셈에 cuSPARSE `cusparseSpMV` 사용하기
5. 희소성 비율에 따라 희소 vs 밀집 표현 선택하기

---

## 1. 희소 표현이 필요한 이유

밀집 행렬은 대부분의 값이 0이더라도 M×N 모든 값을 저장합니다. non-zero가 0.001%인 1,000,000×1,000,000 행렬의 경우, 밀집 저장은 4TB가 필요합니다 — 불가능합니다. 희소 형식은 비-zero 값만 저장합니다.

**희소성 비율**: 0의 비율. 희소성이 ~95% 초과이고 형식 오버헤드가 감수할 만큼 행렬이 충분히 클 때 희소 형식을 사용하세요.

희소 행렬의 일반적인 출처:
- 그래프 인접 행렬 (대형 그래프의 경우 일반적으로 non-zero <0.01%)
- 유한 요소 강성 행렬 (메시 연결성에 의해 제한된 대역폭)
- 가지치기 후 신경망 가중치 행렬 (70–99% zeros)
- 자연어 처리 (단어 공발생, 문서-단어 행렬)

---

## 2. 희소 행렬 형식

### COO — 좌표 형식 (Coordinate Format)

모든 non-zero에 대해 (행, 열, 값) 삼중쌍을 저장합니다:

```c
// COO 표현
typedef struct {
    int    *row_indices;   // [nnz] 각 non-zero의 행 인덱스
    int    *col_indices;   // [nnz] 각 non-zero의 열 인덱스
    float  *values;        // [nnz] 각 non-zero의 값
    int     nrows, ncols, nnz;
} SparseCOO;

// 예: 4×4 행렬, 6개 non-zero
//  [1 0 2 0]
//  [0 3 0 4]
//  [5 0 6 0]
//  [0 7 0 8]
//
// row_indices: [0, 0, 1, 1, 2, 2, 3, 3]
// col_indices: [0, 2, 1, 3, 0, 2, 1, 3]
// values:      [1, 2, 3, 4, 5, 6, 7, 8]
```

COO는 점진적으로 구성하기 쉽지만 SpMV에는 느립니다(행이 연속적이지 않음).

### CSR — 압축 희소 행 (Compressed Sparse Row)

행 인덱스를 포인터 배열로 압축합니다:

```c
typedef struct {
    int    *row_ptr;    // [nrows+1] row_ptr[i] = 행 i의 첫 nnz 인덱스
    int    *col_idx;    // [nnz] 각 non-zero의 열 인덱스
    float  *values;     // [nnz] 각 non-zero의 값
    int     nrows, ncols, nnz;
} SparseCSR;

// 동일한 예를 CSR로:
// row_ptr:  [0, 2, 4, 6, 8]  (행 i의 nnz는 row_ptr[i]부터 row_ptr[i+1]-1까지)
// col_idx:  [0, 2, 1, 3, 0, 2, 1, 3]
// values:   [1, 2, 3, 4, 5, 6, 7, 8]
//
// 행 0: 열 [0,2] 값 [1,2]
// 행 1: 열 [1,3] 값 [3,4]
// 행 2: 열 [0,2] 값 [5,6]
// 행 3: 열 [1,3] 값 [7,8]
```

CSR은 연속 메모리 레이아웃으로 O(nnz_in_row) 행 접근을 가능하게 합니다 — SpMV에 이상적입니다.

### CSC — 압축 희소 열 (Compressed Sparse Column)

CSC는 전치된 CSR입니다 — 행 대신 열에 대한 포인터. 열 방향 접근이 필요할 때 사용됩니다(예: 열 주요 행렬의 SpMM, 또는 A^T × x 직접 계산):

```c
typedef struct {
    int    *col_ptr;    // [ncols+1]
    int    *row_idx;    // [nnz]
    float  *values;     // [nnz]
    int     nrows, ncols, nnz;
} SparseCSC;
// A의 CSC = A^T의 CSR (동일한 메모리 레이아웃, 다른 해석)
```

---

## 3. SpMV: 행당 하나의 Thread (나이브 CSR)

희소 행렬-벡터 곱셈: y = A × x, 여기서 A는 M×N 희소, x는 N-벡터.

```c
// 나이브 SpMV: A의 행당 하나의 thread
__global__ void spmv_csr_scalar(
    const int   *row_ptr,    // [M+1]
    const int   *col_idx,    // [nnz]
    const float *values,     // [nnz]
    const float *x,          // [N]
    float       *y,          // [M] 출력
    int          M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        sum += values[j] * x[col_idx[j]];
    }
    y[row] = sum;
}
```

**문제 — 부하 불균형**: 행 길이가 크게 다양할 경우(예: 그래프: 일부 노드 차수 1, 다른 노드 차수 10,000), warp 내 thread들이 가장 긴 행을 기다리며 지연됩니다. warp는 최대 행 길이만큼 반복을 실행합니다.

---

## 4. SpMV: Warp Per Row

warp의 32개 thread 모두를 단일 행에 할당합니다. 이들이 협력하여 행과 x의 내적을 계산한 후 warp shuffle로 reduce합니다:

```c
// Warp-per-row SpMV — 불규칙 행렬에 대한 더 나은 부하 균형
__global__ void spmv_csr_warp(
    const int   *row_ptr,
    const int   *col_idx,
    const float *values,
    const float *x,
    float       *y,
    int          M)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x & 31;

    if (warp_id >= M) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    float sum = 0.0f;
    // 각 lane이 행의 32번째마다 하나의 요소를 처리
    for (int j = row_start + lane; j < row_end; j += 32) {
        sum += values[j] * x[col_idx[j]];
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) y[warp_id] = sum;
}

// 실행: blockDim.x = 256 (block당 8개 warp), 각각 하나의 행 처리
// grid = (M * 32 + 255) / 256
```

**warp-per-row를 선호할 때**: 행당 평균 ≥ 32개 non-zero일 때. 매우 짧은 행(< 8 nnz)의 경우 warp가 대부분 유휴 상태입니다 — 스칼라 thread 방식을 사용하거나 행 길이에 따라 가변 폭 SIMD 그룹을 행에 할당하는 벡터 방식을 사용하세요.

---

## 5. cuSPARSE SpMV

프로덕션 코드에서는 cuSPARSE를 사용하세요 — 모든 형식 변환, 알고리즘 선택, 자동 튜닝을 처리합니다:

```c
#include <cusparse.h>

void cusparse_spmv(
    const int *h_row_ptr, const int *h_col_idx, const float *h_values,
    const float *h_x, float *h_y,
    int M, int N, int nnz)
{
    // handle 생성
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // device 메모리 할당 및 업로드
    int   *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;
    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz     * sizeof(int));
    cudaMalloc(&d_values,  nnz     * sizeof(float));
    cudaMalloc(&d_x,       N       * sizeof(float));
    cudaMalloc(&d_y,       M       * sizeof(float));
    cudaMemcpy(d_row_ptr, h_row_ptr, (M+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, nnz*sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,  h_values,  nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,       h_x,       N*sizeof(float),   cudaMemcpyHostToDevice);

    // 행렬 및 벡터 디스크립터 생성
    cusparseSpMatDescr_t mat_A;
    cusparseDnVecDescr_t vec_x, vec_y;

    cusparseCreateCsr(&mat_A, M, N, nnz,
                      d_row_ptr, d_col_idx, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vec_x, N, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vec_y, M, d_y, CUDA_R_32F);

    // 버퍼 크기 조회
    float alpha = 1.0f, beta = 0.0f;
    size_t buf_size = 0;
    cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_A, vec_x, &beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);

    void *d_buf;
    cudaMalloc(&d_buf, buf_size);

    // SpMV 실행: y = alpha * A * x + beta * y
    cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_A, vec_x, &beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);

    cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    // 정리
    cusparseDestroySpMat(mat_A);
    cusparseDestroyDnVec(vec_x); cusparseDestroyDnVec(vec_y);
    cudaFree(d_buf); cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    cusparseDestroy(handle);
}
```

---

## 6. SpGEMM 개념 (희소 × 희소)

희소 행렬-행렬 곱셈 (C = A × B, 두 행렬 모두 희소)은 C의 희소성 패턴을 사전에 알 수 없기 때문에 SpMV보다 훨씬 복잡합니다.

cuSPARSE 접근 방식:
1. **작업 추정**: C의 non-zero 수 결정
2. **C 할당**: 추정된 수에 따라
3. **C 계산**: non-zero 값 채우기

```c
// 개념적 cuSPARSE SpGEMM (단순화된 API 스케치)
cusparseSpGEMMDescr_t spgemm_descr;
cusparseSpGEMM_createDescr(&spgemm_descr);

// 단계 1: 작업 추정
cusparseSpGEMM_workEstimation(handle, opA, opB,
    &alpha, mat_A, mat_B, &beta, mat_C,
    CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
    spgemm_descr, &buf1_size, NULL);
cudaMalloc(&d_buf1, buf1_size);

// 단계 2: 계산
cusparseSpGEMM_compute(handle, opA, opB,
    &alpha, mat_A, mat_B, &beta, mat_C,
    CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
    spgemm_descr, &buf2_size, NULL);
// ... (buf2 할당, 결과 복사)
```

---

## 7. 희소 vs 밀집 결정 가이드

```
희소성     행렬 크기          권장 사항
-----------------------------------------------------------
< 90%      모든 크기          밀집 (cuBLAS): 간접 참조 오버헤드가 적음
90–99%     < 10K × 10K        밀집이 여전히 유리할 수 있음 (cuBLAS는 높은 flop/byte)
> 99%      > 100K × 100K      희소 (cuSPARSE): 메모리가 지배적 제약
> 99.9%    > 1M × 1M          희소 필수 (밀집 = 테라바이트)

경험 법칙: nnz/M/N < 0.01 (1% 밀도)이면 희소 사용

특수 케이스:
  구조적 희소성 (블록-희소): cuSPARSE의 block-CSR 또는 BSRMM 사용
  가지치기된 신경망: cuSPARSE 또는 NVIDIA ASP (Accelerated Sparse Precision) 사용
  동적 희소성: COO 또는 hash-map 기반 (변경 시 CSR 재구성)
```

---

## 핵심 요약

- **COO**는 (행, 열, 값) 삼중쌍을 저장합니다 — 구성하기 쉽고 계산에는 느립니다
- **CSR**은 행 포인터를 압축합니다 — SpMV와 행 방향 접근에 최적
- **CSC**는 열 포인터를 압축합니다 — 열 방향 접근에 최적 (또는 A^T × x)
- 행당 하나의 thread SpMV는 **부하 불균형**으로 어려움을 겪습니다; warp-per-row는 ≥32 nnz인 행에서 이를 줄입니다
- **cuSPARSE `cusparseSpMV`** (제네릭 API)는 CSR/CSC/COO를 처리하고 최적의 kernel을 자동으로 선택합니다
- 희소성 > 99%이고 행렬이 크면 희소 형식을 사용하세요; 그 이하에서는 오버헤드가 낮은 밀집 cuBLAS가 종종 더 빠릅니다

---

**다음**: [20. N-Body Simulation](./20_N_Body_Simulation.md) — O(N²) 직접 합산과 tile 기반 shared memory 최적화로 N 입자에 대한 중력 계산을 구현합니다.
