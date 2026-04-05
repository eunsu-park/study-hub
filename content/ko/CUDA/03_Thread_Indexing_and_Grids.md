# 03. 스레드 인덱싱과 그리드

**이전**: [CUDA 프로그래밍 모델](./02_CUDA_Programming_Model.md) | **다음**: [CUDA 메모리 모델](./04_CUDA_Memory_Model.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 1D, 2D, 3D 그리드에 대한 올바른 전역 스레드 인덱스 계산
2. 경계 검사로 임의의 배열 크기 처리
3. 2D 인덱싱을 사용한 행렬 전치 커널 구현
4. 임의의 문제 형태에 맞는 적절한 그리드 및 블록 차원 선택
5. 체계적인 테스트 패턴으로 잘못된 인덱싱 디버깅

---

## 1. 인덱싱이 CUDA 버그 1위인 이유

가장 흔한 CUDA 버그: **잘못된 인덱스 계산**. 1 차이 오류, 누락된 경계 검사, 행/열 순서 혼동은 조용한 오류 결과나 랜덤 충돌을 일으킵니다. 실제 커널을 작성하기 전에 이것을 완전히 익혀야 합니다.

목표: 각 스레드를 처리해야 할 데이터 원소에 정확히 매핑하는 것입니다.

---

## 2. 1D 인덱싱

평탄한 배열의 경우:

```c
__global__ void kernel_1d(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = process(data[i]);
    }
}
```

### N이 blockSize의 배수가 아닐 때

```
N = 10, blockSize = 4:
  gridSize = ceil(10/4) = 3 블록 → 총 12 스레드

블록 0: 스레드 0,1,2,3  → i = 0,1,2,3   ✓ 모두 유효
블록 1: 스레드 0,1,2,3  → i = 4,5,6,7   ✓ 모두 유효
블록 2: 스레드 0,1,2,3  → i = 8,9,10,11 ⚠ i=10,11은 범위 밖 → 경계 검사 필요
```

그리드 크기 공식: `int gridSize = (N + blockSize - 1) / blockSize;`

이는 `ceil(N / blockSize)`와 동일하지만 부동소수점 나눗셈을 피합니다.

---

## 3. 2D 인덱싱

2D 문제 (행렬, 이미지)의 경우, 그리드와 블록 모두 `dim3` 사용:

```c
__global__ void kernel_2d(float *data, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x → 열
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // y → 행

    if (row < rows && col < cols) {
        int idx = row * cols + col;  // 행 우선 (C 관례)
        data[idx] = process(data[idx]);
    }
}

// 실행:
dim3 block(16, 16);  // 블록당 256 스레드, 16×16 배치
dim3 grid(
    (cols + block.x - 1) / block.x,
    (rows + block.y - 1) / block.y
);
kernel_2d<<<grid, block>>>(d_data, rows, cols);
```

**관례**: `x`는 빠르게 변하는 차원(행 우선 레이아웃에서 열)을 인덱싱합니다. 이는 메모리 레이아웃과 일치하여 합치기(coalescing)를 최대화합니다 — 인접한 스레드(같은 행, 인접한 x)가 인접한 메모리 주소에 접근합니다.

```
5×8 행렬에서 스레드 (row=2, col=3):
  blockIdx = (0, 0), blockDim = (8, 4), threadIdx = (3, 2)
  col = 0*8 + 3 = 3
  row = 0*4 + 2 = 2
  idx = 2 * 8 + 3 = 19   ✓
```

---

## 4. 3D 인덱싱

볼륨 (3D 텐서, 복셀 그리드, 배치 연산)의 경우:

```c
__global__ void kernel_3d(float *data, int D, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 너비
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 높이
    int z = blockIdx.z * blockDim.z + threadIdx.z;  // 깊이

    if (x < W && y < H && z < D) {
        int idx = z * (H * W) + y * W + x;
        data[idx] = process(data[idx]);
    }
}

dim3 block(8, 8, 4);  // 3D로 블록당 256 스레드
dim3 grid(
    (W + block.x - 1) / block.x,
    (H + block.y - 1) / block.y,
    (D + block.z - 1) / block.z
);
```

**제한**: `gridDim.z`의 최대값은 65,535입니다. 대형 배치 처리의 경우, 커널 내에서 루프를 사용하거나 하나의 차원에 배치 인덱스를 넣어 2D 그리드를 사용하세요.

---

## 5. 사례 연구: 행렬 전치

행렬 전치는 A[row][col]을 읽어 B[col][row]에 씁니다. 단순한 버전은 문제가 있습니다: 읽기 또는 쓰기 중 하나가 비합치됩니다.

### 단순 버전 (비합치 쓰기)

```c
__global__ void transpose_naive(const float *in, float *out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
        //   ↑ 스트라이드 쓰기: 스레드 0→0, 스레드 1→rows, 스레드 2→2*rows ...
    }
}
```

- `in[row * cols + col]` **읽기**: 합치됨 ✓ (인접 스레드가 인접 메모리 읽기)
- `out[col * rows + row]` **쓰기**: 스트라이드 ✗ (인접 스레드가 stride = rows로 쓰기)

이것은 전형적인 성능 문제입니다 — L08(메모리 합치기)에서 공유 메모리를 스테이징 버퍼로 사용하는 해결책을 다룹니다.

### 타일 버전 (공유 메모리 사용 — 미리 보기)

```c
#define TILE 32

__global__ void transpose_tiled(const float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];  // +1로 뱅크 충돌 방지

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // 타일 읽기: 전역 메모리에서 합치된 읽기
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];

    __syncthreads();  // 모든 스레드가 타일을 채울 때까지 대기

    // 타일을 전치하여 쓰기: 전역 메모리에 합치된 쓰기
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < rows && y < cols)
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}
```

이는 거의 최대 메모리 대역폭을 달성합니다. L05에서 자세히 분석합니다.

---

## 6. 블록 차원 선택

1D 커널의 가이드라인:

| 블록 크기 | 동작 |
|---------|------|
| < 32 | warp 낭비 (32 스레드는 항상 함께 실행) |
| 32 | 최소 — 블록당 warp 1개만, 낮은 점유율 |
| 128 | 일반적인 선택 — 블록당 4 warp, 좋은 점유율 |
| **256** | **가장 일반적 — 8 warp, 좋은 점유율, 광범위한 호환성** |
| 512 | 적절 — 레지스터/공유 메모리 한도 확인 필요 |
| 1024 | 최대 — 레지스터/공유 메모리가 매우 낮을 때만 |

2D 커널의 경우, `(16, 16)`이 표준입니다 — 타일 크기에 맞게 배치된 256 스레드.

```c
// 점유율 계산기 (CUDA 6.5+)
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,           // 권장 그리드 크기
    &blockSize,             // 최적 블록 크기
    myKernel,               // 커널 함수
    0,                      // 블록당 동적 공유 메모리
    0                       // 블록 크기 한도 (0 = 제한 없음)
);
printf("최적 블록 크기: %d\n", blockSize);
```

---

## 7. 임의 크기를 위한 스트라이드 기반 인덱싱

하나의 커널 호출이 그리드가 수용할 수 있는 것보다 더 많은 작업을 처리해야 할 때, **그리드-스트라이드 루프** 사용:

```c
__global__ void scale_stride(float *data, float scalar, long n) {
    // 이 스레드의 첫 번째 원소에서 시작
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    // 그리드의 총 스레드 수만큼 스트라이드
    long stride = (long)gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        data[i] *= scalar;
    }
}

// 임의 그리드 크기로 실행 가능:
scale_stride<<<1024, 256>>>(d_data, 2.0f, 1e9);  // 10억 개 원소
```

장점:
- N에 관계없이 올바르게 작동
- 점유율에 맞게 조정된 고정 크기 그리드 실행 가능 (데이터 크기와 독립적)
- 간단한 디버깅: `<<<1, 1>>>`으로 루프 로직 테스트 가능

---

## 8. 인덱스 오류 디버깅

체계적인 접근법: 인덱스를 검증하는 커널 작성.

```c
__global__ void verify_indexing(int *out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int expected_flat = row * cols + col;
        out[expected_flat] = expected_flat;  // 각 스레드가 자신의 인덱스 기록
    }
}

// 실행 후: out[i] == i인지 모든 i에 대해 확인
// 불일치가 있으면 인덱스 계산 버그를 드러냄
```

경계 1 차이 버그의 경우:

```c
// 센티넬 접근법: -1로 채운 후 -1이 남지 않는지 확인
cudaMemset(d_out, -1, bytes);
myKernel<<<grid, block>>>(d_out, N);
cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
for (int i = 0; i < N; i++) {
    assert(h_out[i] != -1);  // 일부 원소가 기록되지 않음
}
```

---

## 9. 요약: 문제 유형별 인덱스 패턴

```c
// 길이 N의 1D 배열
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) { ... }

// M×N (행 × 열) 행렬, 행 우선
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < M && col < N) {
    int idx = row * N + col;
    ...
}

// 배치 2D: 배치 크기 B, 각각 M×N
int col   = blockIdx.x * blockDim.x + threadIdx.x;
int row   = blockIdx.y * blockDim.y + threadIdx.y;
int batch = blockIdx.z;
if (batch < B && row < M && col < N) {
    int idx = batch * (M * N) + row * N + col;
    ...
}

// 그리드-스트라이드 루프
for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += (long)gridDim.x * blockDim.x) {
    ...
}
```

---

## 핵심 요약

- **1D 인덱스**: `i = blockIdx.x * blockDim.x + threadIdx.x` — 항상 경계 검사 `if (i < N)` 추가
- **2D 관례**: `x` → 열 (빠르게 변함), `y` → 행; 합치기를 위해 행 우선 메모리와 일치
- **블록 크기**: 32의 배수; 256이 안전한 기본값; 조정을 위해 `cudaOccupancyMaxPotentialBlockSize` 사용
- **그리드-스트라이드 루프**: 문제 크기를 그리드 크기에서 분리 — 대형 배열이나 재사용 가능한 커널에 사용
- 결과를 신뢰하기 전에 항상 센티넬 또는 인덱스 확인 커널로 인덱싱 검증

---

**다음**: [04. CUDA 메모리 모델](./04_CUDA_Memory_Model.md) — 전역, 공유, 레지스터, L1/L2, 상수, 텍스처 메모리를 포함한 모든 GPU 메모리 유형 탐색.
