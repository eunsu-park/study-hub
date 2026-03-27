# 03. 텐서 연산과 BLAS

**이전**: [메모리 레이아웃과 스트라이드](./02_Memory_Layout_and_Strides.md) | **다음**: [최적화된 행렬 곱셈](./04_Optimized_Matmul.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 평면 float 배열에 대한 element-wise 연산(add, mul, ReLU, GELU, SiLU) 구현
2. 임의 축에 대한 리덕션 연산(sum, max, mean) 구현
3. 올바른 naive 행렬 곱셈 작성 및 FLOPs 분석
4. 고성능 GEMM을 위한 OpenBLAS `cblas_sgemm` 호출
5. naive vs OpenBLAS matmul 벤치마크 및 성능 차이 설명

---

## 1. Element-Wise 연산

모든 element-wise 연산은 `numel`개의 원소를 순회하며 스칼라 함수를 적용합니다:

```c
// ops.h
#pragma once
#include "tensor.h"

// Element-wise 이진 연산 (in-place: out = a OP b, 브로드캐스트 미처리)
void tensor_add(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_mul(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_sub(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_div(Tensor *out, const Tensor *a, const Tensor *b);

// 스칼라 연산
void tensor_add_scalar(Tensor *out, const Tensor *a, float scalar);
void tensor_mul_scalar(Tensor *out, const Tensor *a, float scalar);

// 활성화 함수
void tensor_relu   (Tensor *out, const Tensor *x);
void tensor_gelu   (Tensor *out, const Tensor *x);  // GPT-2 활성화
void tensor_silu   (Tensor *out, const Tensor *x);  // Llama 활성화 (sigmoid * x)
void tensor_sigmoid(Tensor *out, const Tensor *x);

// 리덕션
float  tensor_sum  (const Tensor *x);
float  tensor_max  (const Tensor *x);
float  tensor_mean (const Tensor *x);
Tensor *tensor_sum_axis (const Tensor *x, int axis, bool keepdim);

// 행렬 곱셈
void tensor_matmul      (Tensor *out, const Tensor *a, const Tensor *b);  // naive
void tensor_matmul_blas (Tensor *out, const Tensor *a, const Tensor *b);  // OpenBLAS
```

### 구현: Element-Wise

```c
// ops.c
#include "ops.h"
#include <math.h>

void tensor_add(Tensor *out, const Tensor *a, const Tensor *b) {
    assert(a->numel == b->numel && a->numel == out->numel);
    for (size_t i = 0; i < a->numel; i++)
        out->data[i] = a->data[i] + b->data[i];
}

void tensor_relu(Tensor *out, const Tensor *x) {
    for (size_t i = 0; i < x->numel; i++)
        out->data[i] = x->data[i] > 0.0f ? x->data[i] : 0.0f;
}
```

### GELU와 SiLU

```c
#include <math.h>

// GELU: GPT-2 FFN에서 사용
// 근사값: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
void tensor_gelu(Tensor *out, const Tensor *x) {
    const float sqrt2_over_pi = 0.7978845608f;  // sqrt(2/π)
    const float coef = 0.044715f;
    for (size_t i = 0; i < x->numel; i++) {
        float v = x->data[i];
        float inner = sqrt2_over_pi * (v + coef * v * v * v);
        out->data[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

// SiLU (Swish): Llama FFN에서 사용
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
void tensor_silu(Tensor *out, const Tensor *x) {
    for (size_t i = 0; i < x->numel; i++) {
        float v = x->data[i];
        out->data[i] = v / (1.0f + expf(-v));
    }
}
```

---

## 2. 리덕션

리덕션은 전체 또는 한 차원의 원소를 집계합니다.

```c
float tensor_sum(const Tensor *x) {
    float acc = 0.0f;
    for (size_t i = 0; i < x->numel; i++) acc += x->data[i];
    return acc;
}

float tensor_max(const Tensor *x) {
    assert(x->numel > 0);
    float m = x->data[0];
    for (size_t i = 1; i < x->numel; i++)
        if (x->data[i] > m) m = x->data[i];
    return m;
}

float tensor_mean(const Tensor *x) {
    return tensor_sum(x) / (float)x->numel;
}
```

### 축 리덕션

shape `[M, N]`인 2D 텐서의 축 `ax` 리덕션:

| 축 | 연산 | 출력 shape |
|----|------|-----------|
| 0 | 행 합산 | `[N]` |
| 1 | 열 합산 | `[M]` |

```c
// 2D 전용 축 리덕션 (이후 레슨에서 일반화)
Tensor *tensor_sum_axis2d(const Tensor *x, int axis) {
    assert(x->ndim == 2);
    size_t M = x->shape[0], N = x->shape[1];

    if (axis == 0) {
        size_t out_shape[] = {N};
        Tensor *out = tensor_zeros(1, out_shape);
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                out->data[j] += x->data[i * N + j];
        return out;
    } else {  // axis == 1
        size_t out_shape[] = {M};
        Tensor *out = tensor_zeros(1, out_shape);
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                out->data[i] += x->data[i * N + j];
        return out;
    }
}
```

---

## 3. Naive 행렬 곱셈

shape `A[M, K]`, `B[K, N]`, `C[M, N]`에 대한 행렬 곱셈 `C = A * B`:

```
C[i][j] = sum_{k=0}^{K-1} A[i][k] * B[k][j]
```

```c
// Naive 3중 루프 matmul — 정확하지만 느림
void tensor_matmul_naive(Tensor *C, const Tensor *A, const Tensor *B) {
    assert(A->ndim == 2 && B->ndim == 2 && C->ndim == 2);
    size_t M = A->shape[0], K = A->shape[1], N = B->shape[1];
    assert(B->shape[0] == K && C->shape[0] == M && C->shape[1] == N);

    memset(C->data, 0, M * N * sizeof(float));

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < K; k++)
                C->data[i * N + j] += A->data[i * K + k] * B->data[k * N + j];
}
```

### FLOPs 분석

`C = A[M,K] * B[K,N]`의 경우:
- 각 `C[i,j]`는 `K`번의 multiply-add 필요
- 총 FLOPs = `2 * M * K * N` (내적 스텝당 곱셈 하나 + 덧셈 하나)

GPT-2의 attention Q 투영 `[batch * seq, d_model] * [d_model, d_head]`의 경우:
- `M = 512`, `K = 768`, `N = 64`
- FLOPs = `2 * 512 * 768 * 64 ≈ 5천만` 배치 항목당 헤드당

---

## 4. BLAS: 기본 선형 대수 서브프로그램

OpenBLAS는 SIMD와 멀티스레딩을 사용하여 고도로 최적화된 GEMM(일반 행렬-행렬 곱셈)을 제공합니다. 표준 인터페이스는 `cblas_sgemm`입니다.

### CBLAS SGEMM 시그니처

```c
void cblas_sgemm(
    CBLAS_LAYOUT    layout,    // CblasRowMajor 또는 CblasColMajor
    CBLAS_TRANSPOSE TransA,    // CblasNoTrans 또는 CblasTrans
    CBLAS_TRANSPOSE TransB,
    int             M,         // A와 C의 행 수
    int             N,         // B와 C의 열 수
    int             K,         // A의 열 수, B의 행 수
    float           alpha,     // 스칼라 배수: C = alpha*A*B + beta*C
    const float    *A,
    int             lda,       // A의 leading dimension (행 간 stride)
    const float    *B,
    int             ldb,
    float           beta,      // C 누적을 위한 스칼라
    float          *C,
    int             ldc
);
```

### 래퍼

```c
#include <cblas.h>

void tensor_matmul_blas(Tensor *C, const Tensor *A, const Tensor *B) {
    assert(A->ndim == 2 && B->ndim == 2 && C->ndim == 2);
    int M = (int)A->shape[0];
    int K = (int)A->shape[1];
    int N = (int)B->shape[1];
    assert((int)B->shape[0] == K);
    assert((int)C->shape[0] == M && (int)C->shape[1] == N);

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0f,           // alpha
        A->data, K,     // A, lda = K (행 우선의 행 stride)
        B->data, N,     // B, ldb = N
        0.0f,           // beta (C 덮어쓰기)
        C->data, N      // C, ldc = N
    );
}
```

---

## 5. 벤치마크: Naive vs OpenBLAS

```c
// benchmark_matmul.c
#include <time.h>
#include <stdio.h>
#include "tensor.h"
#include "ops.h"

double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    int sizes[] = {128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        size_t N = sizes[s];
        size_t shape[] = {N, N};
        Tensor *A = tensor_zeros(2, shape);
        Tensor *B = tensor_zeros(2, shape);
        Tensor *C = tensor_zeros(2, shape);

        for (size_t i = 0; i < N * N; i++) {
            A->data[i] = (float)rand() / RAND_MAX;
            B->data[i] = (float)rand() / RAND_MAX;
        }

        double flops = 2.0 * N * N * N;

        // BLAS matmul
        double t2 = get_time_ms();
        tensor_matmul_blas(C, A, B);
        double t3 = get_time_ms();

        double blas_gflops = flops / (t3 - t2) / 1e6;
        printf("N=%4zu  BLAS: %6.1f GFLOP/s  (%5.1f ms)\n",
               N, blas_gflops, t3 - t2);

        tensor_free(A); tensor_free(B); tensor_free(C);
    }
    return 0;
}
```

**현대 CPU에서의 일반적인 결과 (단일 스레드 OpenBLAS, AVX2)**:

```
N= 128  BLAS:  120 GFLOP/s   (0.0 ms)  Naive:   0.5 ms  속도향상:  25x
N= 256  BLAS:  180 GFLOP/s   (0.2 ms)  Naive:   3.2 ms  속도향상:  16x
N= 512  BLAS:  210 GFLOP/s   (1.3 ms)  Naive:  55.0 ms  속도향상:  42x
N=1024  BLAS:  230 GFLOP/s  (11.0 ms)
N=2048  BLAS:  240 GFLOP/s  (85.0 ms)
```

차이가 큰 이유는 OpenBLAS가 다음을 사용하기 때문입니다:
1. **AVX2/AVX-512** — 명령당 8개 또는 16개의 float
2. **루프 타일링** — 데이터를 L1/L2 캐시에 유지
3. **멀티스레딩** — 모든 코어 활용

L04에서 이 최적화들을 직접 구현합니다.

---

## 6. FLOP/Byte 비율 (산술 강도)

성능 병목을 이해하기 위한 핵심 개념:

```
산술 강도 = FLOPs / 메모리에서 읽은 바이트 수

Naive matmul N×N:
  FLOPs  = 2 * N^3
  Bytes  = 3 * N^2 * 4  (A, B 읽기; C 쓰기)
  AI     = 2N^3 / (12N^2) = N/6

N=1024의 경우: AI ≈ 170 FLOPs/byte
현대 CPU: ~24 GFLOP/s (AVX2 단일 스레드), ~50 GB/s 메모리 대역폭
  → 연산 한계 임계값: 24e9 / 50e9 = 0.48 FLOPs/byte
  → AI=170 >> 0.48: matmul은 강하게 연산 한계
  → 메모리 대역폭이 아닌 처리량을 개선할 여지가 있음
```

이것이 **roofline 모델**이다 — 연산 바운드 vs. 메모리 바운드 연산을 분석하는 데 유용한 프레임워크이다.

---

## 핵심 요약

- Element-wise 연산은 자명하게 병렬화 가능합니다; 루프 본문은 단순한 스칼라 함수
- GELU는 `tanh`(비용 높음)를 사용하고; SiLU는 `sigmoid`(비용 낮음)를 사용합니다 — Llama의 SwiGLU는 이들 두 개를 사용
- Naive matmul은 `O(N^3)`이고 큰 N에서 캐시 동작이 매우 나쁩니다
- OpenBLAS는 SIMD + 타일링 + 스레딩을 통해 피크에 가까운 FLOP/s를 달성합니다
- 산술 강도는 커널이 메모리 한계인지 연산 한계인지를 결정합니다

---

**다음**: [04. 최적화된 행렬 곱셈](./04_Optimized_Matmul.md) — 루프 타일링과 AVX2 intrinsics를 구현하여 OpenBLAS 성능에 근접한 SGEMM을 만듭니다.
