# 02. CUDA 프로그래밍 모델

**이전**: [GPU 아키텍처 개요](./01_GPU_Architecture_Overview.md) | **다음**: [스레드 인덱싱과 그리드](./03_Thread_Indexing_and_Grids.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CUDA 스레드 계층 설명: 스레드 → warp → 블록 → 그리드
2. `<<<grid, block>>>` 문법으로 CUDA 커널 작성 및 실행
3. `cudaMalloc`, `cudaFree`, `cudaMemcpy`로 메모리 할당 및 전송
4. `vector_add.cu` 구현 — GPU 컴퓨팅의 "Hello World"
5. `cudaGetLastError`로 오류 처리 및 오류 메시지 해석

---

## 1. CUDA 실행 모델

CUDA 프로그램은 두 프로세서에서 동시에 실행됩니다:

```
호스트 (CPU)                         장치 (GPU)
─────────────────                    ────────────────────────────────
순차적 C/C++ 코드                    수천 개의 병렬 스레드
커널 실행 (launch)                   커널 실행 (execute)
데이터 전송 관리                     자체 메모리 공간 보유
```

**커널(kernel)**은 GPU에서 실행되는 C 함수입니다. 모든 스레드가 같은 커널 코드를 실행하지만, 고유한 스레드 인덱스로 식별된 서로 다른 데이터를 처리합니다.

---

## 2. 스레드 계층 구조

CUDA는 스레드를 3단계 계층으로 구성합니다:

```
그리드 (커널 호출 하나에서 실행되는 모든 스레드)
│
├── 블록 0          블록 1          블록 2 ...
│   ├── 스레드 0    ├── 스레드 0    ├── 스레드 0
│   ├── 스레드 1    ├── 스레드 1    ├── 스레드 1
│   ├── 스레드 2    ├── 스레드 2    ├── 스레드 2
│   └── ...         └── ...         └── ...
```

| 단계 | 변수 | 범위 | 비고 |
|------|------|------|------|
| **스레드** | `threadIdx.{x,y,z}` | 블록 내 | 블록당 최대 1024 스레드 |
| **블록** | `blockIdx.{x,y,z}` | 그리드 내 | 하나의 SM에서 실행 |
| **그리드** | `gridDim.{x,y,z}` | 전체 커널 | 최대 2³¹−1 블록 |

**핵심 제약**: 하나의 블록은 **하나의 SM**에서 완전히 실행되며 분할될 수 없습니다. 같은 블록 내 스레드는 **공유 메모리**와 `__syncthreads()`를 통해 협력할 수 있습니다. 서로 다른 블록의 스레드는 직접 통신할 수 없습니다.

---

## 3. `<<<grid, block>>>` 실행 문법

```c
kernel<<<gridDim, blockDim>>>(인자...);
```

- `gridDim`: 그리드의 블록 수 (`dim3` 또는 `int`)
- `blockDim`: 블록당 스레드 수 (`dim3` 또는 `int`)

```c
// 1D 예시: 100만 개 원소
int N = 1 << 20;            // 1,048,576개 원소
int blockSize = 256;        // 블록당 스레드 수 (32의 배수여야 함)
int gridSize  = (N + blockSize - 1) / blockSize;  // = 4096 블록

myKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
```

2D 문제 (예: 행렬):

```c
dim3 block(16, 16);           // 블록당 256 스레드, 2D 배치
dim3 grid(W/16, H/16);        // 16×16 타일당 하나의 블록
matmulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
```

---

## 4. 메모리 관리

GPU 메모리는 CPU 메모리와 분리되어 있습니다. 명시적으로 할당하고 전송해야 합니다:

```c
// 호스트 (CPU) 할당
float *h_a = (float *)malloc(N * sizeof(float));

// 장치 (GPU) 할당
float *d_a;
cudaMalloc((void **)&d_a, N * sizeof(float));

// 전송: 호스트 → 장치
cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

// ... GPU에서 커널 실행 ...

// 전송: 장치 → 호스트
cudaMemcpy(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);

// GPU 메모리 해제
cudaFree(d_a);
free(h_a);
```

### 메모리 전송 비용

```
PCI-e 4.0 ×16:  양방향 ~32 GB/s
NVLink (A100):  GPU–GPU ~600 GB/s

PCIe로 1 GB 전송: ~31 ms
HBM2e 대역폭 (A100): 2 TB/s — PCIe보다 62배 빠름
```

**경험 법칙**: 호스트↔장치 전송을 최소화하세요. 가능한 한 오랫동안 데이터를 GPU에 유지하세요.

---

## 5. 벡터 덧셈 — 첫 번째 CUDA 커널

CUDA의 정석적인 첫 번째 프로그램: 두 배열의 원소별 덧셈.

```c
// vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

// 커널: GPU에서 실행, CPU에서 호출
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 전역 스레드 인덱스
    if (i < n) {          // 경계 검사
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    const int N = 1 << 20;  // 100만 개 원소
    const size_t bytes = N * sizeof(float);

    // 호스트 배열
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // 장치 배열
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 입력 데이터를 GPU로 복사
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 커널 실행
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // 결과 복사
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 검증
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        float err = fabsf(h_c[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("최대 오차: %e\n", max_err);  // 0.000000e+00이어야 함

    // 정리
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

```bash
nvcc -O2 -arch=sm_80 -o vector_add vector_add.cu
./vector_add
# 출력: 최대 오차: 0.000000e+00
```

### 인덱스 계산 작동 방식

```
blockIdx.x = 2, blockDim.x = 256, threadIdx.x = 37
→ i = 2 * 256 + 37 = 549

각 스레드는 정확히 하나의 원소 처리: c[549] = a[549] + b[549]
```

N=1,048,576 원소, blockSize=256일 때:
- gridSize = 1,048,576 / 256 = **4,096 블록**
- 총 스레드 = 4,096 × 256 = **1,048,576** (원소당 하나)

---

## 6. 함수 한정자

CUDA는 코드가 실행될 위치를 제어하는 한정자를 사용합니다:

| 한정자 | 실행 위치 | 호출 위치 | 비고 |
|--------|---------|---------|------|
| `__global__` | GPU | CPU (또는 CC 3.5+에서 GPU) | 커널 — 주 진입점 |
| `__device__` | GPU | GPU만 | 커널에서 호출되는 헬퍼 함수 |
| `__host__` | CPU | CPU만 | 일반 C 함수 (기본값) |
| `__host__ __device__` | 양쪽 | 양쪽 | 두 대상 모두 컴파일 |

```c
__device__ float square(float x) { return x * x; }

__global__ void squareKernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = square(data[i]);  // __device__ 함수 호출
}
```

---

## 7. 오류 처리

CUDA 함수는 `cudaError_t`를 반환합니다. 프로덕션 코드에서는 항상 오류를 확인하세요:

```c
// CUDA 오류 확인 매크로
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA 오류 at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 사용법
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

// 커널 실행은 오류를 직접 반환하지 않음 — 실행 후 확인
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());         // 실행 오류 포착
CUDA_CHECK(cudaDeviceSynchronize());    // 커널 완료 대기 + 런타임 오류 포착
```

**동기화가 필요한 이유**: GPU 커널은 **비동기적** — CPU는 커널 실행 후 즉시 계속됩니다. `cudaDeviceSynchronize()`는 GPU가 완료될 때까지 CPU를 블록합니다.

### compute-sanitizer로 메모리 오류 디버깅

`CUDA_CHECK`가 잡지 못하는 더 깊은 런타임 오류 감지를 위해 `compute-sanitizer`를 사용하세요:

```bash
# 범위 초과, 초기화되지 않은 메모리, 잘못 정렬된 접근, 이중 해제 감지
compute-sanitizer --tool memcheck ./my_program

# 그 외 사용 가능한 도구:
compute-sanitizer --tool racecheck  ./my_program   # 공유 메모리 레이스 컨디션
compute-sanitizer --tool initcheck  ./my_program   # 초기화되지 않은 전역 메모리 읽기
compute-sanitizer --tool synccheck  ./my_program   # __syncthreads() 오용
```

`compute-sanitizer --tool memcheck`는 CUDA의 Valgrind(메모리 오류 검사 도구)에 해당합니다. 개발 중 모든 새 커널에 대해 실행해야 합니다 — 자동 감지되지 않는 범위 초과 쓰기는 이 도구 없이 진단하기 가장 어려운 CUDA 버그에 속합니다. Sanitizer 실행 중에는 5–20배의 속도 저하를 예상하세요.

---

## 8. CUDA 이벤트로 커널 시간 측정

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);  // stop 이벤트 대기

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("커널 시간: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 9. 내장 변수 치트시트

```c
// __global__ 또는 __device__ 함수 내부:
threadIdx.x / .y / .z    // 블록 내 스레드의 위치
blockIdx.x  / .y / .z    // 그리드 내 블록의 위치
blockDim.x  / .y / .z    // 블록 차원 (실행 시 설정)
gridDim.x   / .y / .z    // 그리드 차원 (실행 시 설정)

// 일반적인 1D 인덱스 패턴:
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 일반적인 2D 인덱스 패턴 (예: 행렬 원소 [row][col]):
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;

// Warp 레인 (0–31):
int lane = threadIdx.x % 32;

// 블록 내 warp ID:
int warpId = threadIdx.x / 32;
```

---

## 10. 완전한 워크플로우 다이어그램

```
CPU (호스트)                        GPU (장치)
──────────────────────────────    ──────────────────────────────────
malloc(h_a, h_b, h_c)
h_a, h_b 초기화
              ──cudaMalloc──→      d_a, d_b, d_c 할당
              ──H2D 복사──→        d_a = h_a, d_b = h_b
커널 <<<>>> 실행                    │
(CPU 즉시 반환)                     ↓
                                   스레드 0: c[0] = a[0]+b[0]
                                   스레드 1: c[1] = a[1]+b[1]
                                   ...
                                   스레드 N-1: c[N-1] = a[N-1]+b[N-1]
cudaDeviceSynchronize() ←─────────  (커널 완료)
              ←──D2H 복사──        h_c = d_c
결과 검증
cudaFree, free
```

---

## 핵심 요약

- **커널**은 `<<<gridDim, blockDim>>>`으로 실행되며 수천 개의 GPU 스레드에서 동시에 실행됨
- 각 스레드는 전역 인덱스 계산: `i = blockIdx.x * blockDim.x + threadIdx.x`
- **블록 크기는 32(warp 크기)의 배수**여야 함; 128 또는 256이 일반적인 최적값
- GPU 메모리는 분리되어 있음 — `cudaMalloc`/`cudaMemcpy`/`cudaFree`로 관리
- 커널 실행은 **비동기적**; 결과에 접근하기 전에 `cudaDeviceSynchronize()` 필요
- 항상 `CUDA_CHECK()`로 오류 확인; 자동 감지되지 않는 CUDA 오류는 흔한 함정

---

**다음**: [03. 스레드 인덱싱과 그리드](./03_Thread_Indexing_and_Grids.md) — 1D/2D/3D 그리드 인덱싱 마스터, 임의 배열 크기 처리, 행렬 전치 커널 구현.
