# 02. 메모리 레이아웃과 스트라이드

**이전**: [왜 딥러닝에 C/C++를 사용하는가?](./01_Why_C_for_DL.md) | **다음**: [텐서 연산과 BLAS](./03_Tensor_Ops_BLAS.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 행 우선(C-order) vs. 열 우선(Fortran-order) 메모리 레이아웃 설명
2. N차원 텐서에 대한 스트라이드 산술로 요소 주소 계산
3. 데이터 복사 없이 reshape, transpose, slice를 수행하는 제로 카피 뷰 구현
4. 비연속 텐서 감지 및 처리
5. 캐시 라인 정렬과 matmul 성능에 미치는 영향 설명

---

## 1. 메모리 레이아웃 기초

텐서는 개념적으로 다차원이지만, 물리적으로는 메모리에서 **평면의 연속적인 float 블록**으로 저장됩니다.

### 행 우선 (C-order)

행 우선 레이아웃에서는 **마지막 인덱스가 가장 빠르게 변합니다**. C, NumPy(기본값), PyTorch의 기본값입니다.

```
행렬 A (2×3):
  A[0][0] A[0][1] A[0][2]
  A[1][0] A[1][1] A[1][2]

메모리 (연속):
  인덱스: 0       1       2       3       4       5
  값:     A[0,0]  A[0,1]  A[0,2]  A[1,0]  A[1,1]  A[1,2]
```

`A[i][j]` 접근:
```
offset = i * 열수 + j
       = i * stride[0] + j * stride[1]
```

shape `[2, 3]`에서 strides는 `[3, 1]`입니다.

### 열 우선 (Fortran-order)

**첫 번째 인덱스가 가장 빠르게 변합니다**. Fortran, MATLAB, cuBLAS 기반 저장소에서 사용됩니다.

```
열 우선 A (2×3):
  offset:  0       1       2       3       4       5
  값:      A[0,0]  A[1,0]  A[0,1]  A[1,1]  A[0,2]  A[1,2]
```

shape `[2, 3]`의 열 우선 strides: `[1, 2]`.

> **왜 중요한가**: cuBLAS는 열 우선 행렬을 기대합니다. `cublasSgemm` 호출 시 행 우선 PyTorch/C 텐서로 작업하려면 종종 연산을 전치해야 합니다. 스트라이드를 이해하면 데이터 복사 없이 cuBLAS를 호출할 수 있습니다.

### 일반 규칙: 스트라이드

shape `[d0, d1, ..., d_{n-1}]`인 **행 우선** 텐서의 경우:
```
stride[n-1] = 1
stride[k]   = stride[k+1] * shape[k+1]   (k = n-2 내림차순 0까지)
```

요소 `t[i0, i1, ..., i_{n-1}]`의 주소:
```
offset  = sum(i_k * stride[k])   (k = 0..n-1)
address = data + offset
```

---

## 2. 텐서 구조체 확장

```c
// tensor.h
#pragma once
#include <stddef.h>
#include <stdbool.h>

#define TENSOR_MAX_DIMS 8

typedef struct Tensor {
    float  *data;                      // 원시 데이터 포인터 (공유 가능)
    size_t  shape[TENSOR_MAX_DIMS];
    size_t  strides[TENSOR_MAX_DIMS];  // 바이트가 아닌 원소 단위
    int     ndim;
    size_t  numel;
    bool    owns_data;                 // false → 뷰 (data를 해제하지 않음)

    // 자동미분 필드 (L05에서 추가)
    struct Tensor *grad;
    void (*backward_fn)(struct Tensor *self);
    void *backward_ctx;
    bool  requires_grad;
} Tensor;

// 할당
Tensor *tensor_zeros(int ndim, const size_t *shape);
Tensor *tensor_ones(int ndim, const size_t *shape);
Tensor *tensor_from_data(float *data, int ndim, const size_t *shape, bool owns);

// 뷰 (제로 카피)
Tensor *tensor_view(Tensor *src, int ndim, const size_t *new_shape);
Tensor *tensor_transpose(Tensor *src, int dim0, int dim1);
Tensor *tensor_slice(Tensor *src, int dim, size_t start, size_t end);

// 속성
bool   tensor_is_contiguous(const Tensor *t);
Tensor *tensor_contiguous(Tensor *t);   // 필요 시 연속 복사본 반환

void   tensor_free(Tensor *t);
void   tensor_print(const Tensor *t, const char *name);
```

---

## 3. 뷰 구현

### Reshape (View)

reshape은 텐서가 연속적일 때만 유효합니다(스트라이드가 표준 행 우선). 동일한 *데이터*를 가리키는 새 `Tensor` 헤더를 만듭니다.

```c
Tensor *tensor_view(Tensor *src, int new_ndim, const size_t *new_shape) {
    // 총 원소 수 일치 확인
    size_t numel = 1;
    for (int i = 0; i < new_ndim; i++) numel *= new_shape[i];
    assert(numel == src->numel && "view: 원소 수 불일치");
    assert(tensor_is_contiguous(src) && "view: 소스는 연속적이어야 함");

    Tensor *t     = (Tensor *)calloc(1, sizeof(Tensor));
    t->data       = src->data;    // 공유 포인터 — 복사 없음!
    t->ndim       = new_ndim;
    t->numel      = numel;
    t->owns_data  = false;        // tensor_free(t) 시 해제하지 않음

    for (int i = 0; i < new_ndim; i++) t->shape[i] = new_shape[i];

    // 새 shape에 대한 행 우선 strides 계산
    t->strides[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--)
        t->strides[i] = t->strides[i + 1] * new_shape[i + 1];

    return t;
}
```

### Transpose (전치)

전치는 두 차원의 strides를 교환합니다 — 데이터 이동 없음.

```c
Tensor *tensor_transpose(Tensor *src, int dim0, int dim1) {
    assert(dim0 < src->ndim && dim1 < src->ndim);

    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    t->data      = src->data;
    t->ndim      = src->ndim;
    t->numel     = src->numel;
    t->owns_data = false;

    memcpy(t->shape,   src->shape,   src->ndim * sizeof(size_t));
    memcpy(t->strides, src->strides, src->ndim * sizeof(size_t));

    // 두 차원의 shape과 stride 교환
    size_t tmp_shape  = t->shape[dim0];
    t->shape[dim0]    = t->shape[dim1];
    t->shape[dim1]    = tmp_shape;

    size_t tmp_stride  = t->strides[dim0];
    t->strides[dim0]   = t->strides[dim1];
    t->strides[dim1]   = tmp_stride;

    return t;
}
```

**예시**: `[4, 3]` 행렬 전치
```
원본:    shape=[4,3], strides=[3,1]
전치 후: shape=[3,4], strides=[1,3]

T[i][j] = data[ i*1 + j*3 ] = data[ j*3 + i ]
         = A[j][i]  ✓
```

---

## 4. 연속성 검사

텐서가 **연속적**이라는 것은 strides가 shape에 대한 표준 행 우선 레이아웃과 일치할 때입니다.

```c
bool tensor_is_contiguous(const Tensor *t) {
    size_t expected = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected) return false;
        expected *= t->shape[i];
    }
    return true;
}
```

비연속 텐서(예: 전치된 텐서)는 `view`로 reshape할 수 없습니다. 표준 strides로 새 버퍼에 데이터를 복사하여 먼저 연속적으로 만들어야 합니다.

```c
Tensor *tensor_contiguous(Tensor *t) {
    if (tensor_is_contiguous(t)) return t;

    Tensor *out = tensor_zeros(t->ndim, t->shape);
    // stride 기반 인덱싱을 사용한 모든 원소 순회
    size_t coords[TENSOR_MAX_DIMS] = {0};
    for (size_t flat = 0; flat < t->numel; flat++) {
        // coords + strides로 소스 offset 계산
        size_t src_offset = 0;
        for (int d = 0; d < t->ndim; d++)
            src_offset += coords[d] * t->strides[d];

        out->data[flat] = t->data[src_offset];

        // coords 증가 (우→좌)
        for (int d = t->ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < t->shape[d]) break;
            coords[d] = 0;
        }
    }
    return out;
}
```

---

## 5. 캐시 라인 정렬

현대 CPU는 **64바이트 캐시 라인**으로 데이터를 로드합니다. `float`는 4바이트이므로 캐시 라인 하나에 16개의 float가 들어갑니다.

```
행렬 A [1024 x 1024]:
  행 접근 (A[i][j], A[i][j+1], ...): 연속적 → 원소 16개당 캐시 미스 1회
  열 접근 (A[0][j], A[1][j], ...):   stride=1024 → 원소당 캐시 미스 1회
```

matmul `C = A * B`에서 `B`를 열 방향으로 접근하면 **캐시 스래싱**이 발생합니다. 해결책은:
1. 곱셈 전에 `B`를 전치 (B^T는 행 방향으로 접근됨)
2. 또는 **타일링** 사용 (L1 캐시에 맞는 B 블록 접근 — L04에서 다룸)

**메모리 정렬**: SIMD 효율성을 위해 64바이트 경계에 데이터 할당:

```c
#include <stdlib.h>

float *alloc_aligned(size_t numel) {
    void *ptr = NULL;
    // posix_memalign은 주어진 경계로의 정렬을 보장
    if (posix_memalign(&ptr, 64, numel * sizeof(float)) != 0) {
        fprintf(stderr, "alloc_aligned: 할당 실패\n");
        exit(1);
    }
    return (float *)ptr;
}
```

---

## 6. 실전: NCHW vs NHWC

합성곱에는 두 가지 일반적인 4D 레이아웃이 있습니다:

| 레이아웃 | Shape | 사용 |
|---------|-------|------|
| **NCHW** | [배치, 채널, 높이, 너비] | PyTorch 기본값, CUDA 선호 |
| **NHWC** | [배치, 높이, 너비, 채널] | TensorFlow 기본값, ARM 선호 |

```
NCHW strides for [N, C, H, W]:
  stride[0] = C * H * W   (이미지 하나를 넘어가는 스텝)
  stride[1] = H * W       (채널 하나를 넘어가는 스텝)
  stride[2] = W           (한 행을 넘어가는 스텝)
  stride[3] = 1           (한 픽셀을 넘어가는 스텝)
```

C에서는 **strides를 변경**하여 레이아웃을 전환할 수 있습니다 — 데이터 복사 불필요. 단, conv 커널은 이웃에 올바르게 접근하기 위해 적절한 stride 공식을 사용해야 합니다.

---

## 7. 실습 연습

### 연습 1: 스트라이드 계산

shape `[3, 4, 5]`인 텐서 `t`(행 우선)의:
- `t.strides` (정답: `[20, 5, 1]`)
- `t[2][1][3]`의 flat 인덱스 (정답: `2*20 + 1*5 + 3*1 = 48`)

### 연습 2: 전치 및 검증

```c
// 3×4 행렬 생성, 순차값으로 채우기
// 4×3 행렬로 전치
// transposed[i][j] == original[j][i] 검증
```

### 연습 3: 비연속 뷰

```c
// [6, 6] 행렬 생성
// 행 1..4 슬라이싱 (뷰, 완전히 연속적이지 않을 수 있음)
// 연속적으로 만들고 데이터 검증
```

---

## 핵심 요약

- 스트라이드는 제로 카피 reshape, transpose, slice를 가능하게 하는 핵심 추상화
- 행 우선(C-order): strides `[d1*d2*...*dn, d2*...*dn, ..., 1]`
- 뷰는 기반 데이터 포인터를 공유합니다 — 헤더(shape, strides)만 변경
- 비연속 텐서(transpose나 slice로 생성된)는 reshape 전에 복사가 필요
- 캐시 라인 정렬(64바이트)과 연속적 접근 패턴은 SIMD 성능에 중요

---

**다음**: [03. 텐서 연산과 BLAS](./03_Tensor_Ops_BLAS.md) — element-wise 연산, 리덕션, naive matmul을 구현하고 OpenBLAS와 벤치마크합니다.
