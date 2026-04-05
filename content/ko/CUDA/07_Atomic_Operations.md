# 07. 원자적 연산

**이전**: [Warp 실행과 분기 발산](./06_Warp_Execution_and_Divergence.md) | **다음**: [메모리 합치기](./08_Memory_Coalescing.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 병렬 코드에서 일반 읽기-수정-쓰기가 안전하지 않은 이유 설명
2. `atomicAdd`, `atomicCAS`, `atomicExch` 및 관련 내장 함수 사용
3. 원자적 연산을 사용하여 올바른 병렬 히스토그램 구현
4. 원자적 처리량 vs 충돌 비용 측정 및 이해
5. 원자적 충돌을 극적으로 줄이기 위한 사유화(privatization) 적용

---

## 1. 경쟁 조건 문제

원자적 연산 없이 같은 주소에 동시에 쓰면 잘못된 결과가 발생합니다:

```c
// 단순한 병렬 카운터 — 잘못됨
__global__ void count_positives(const float *data, int *count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] > 0.0f) {
        (*count)++;  // 경쟁 조건! 읽기-수정-쓰기가 원자적이지 않음
    }
}

// 무슨 일이 일어나는가:
// 스레드 0이 count = 5 읽기
// 스레드 1이 count = 5 읽기  (스레드 0이 쓰기 전)
// 스레드 0이 count = 6 쓰기
// 스레드 1이 count = 6 쓰기  (스레드 0의 결과 덮어씀!)
// 갱신 손실: 양수 값 두 개를 찾았지만 count는 1만 증가
```

`atomicAdd`가 이를 해결합니다:

```c
__global__ void count_positives_atomic(const float *data, int *count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] > 0.0f) {
        atomicAdd(count, 1);  // 원자적 읽기-수정-쓰기 — 항상 올바름
    }
}
```

---

## 2. 원자적 연산 레퍼런스

모든 원자적 연산은 연산 전 대상의 **이전 값**을 반환합니다:

```c
// 정수 원자적 연산
int old = atomicAdd(int *addr, int val);      // *addr += val
int old = atomicSub(int *addr, int val);      // *addr -= val
int old = atomicMax(int *addr, int val);      // *addr = max(*addr, val)
int old = atomicMin(int *addr, int val);      // *addr = min(*addr, val)
int old = atomicAnd(int *addr, int val);      // *addr &= val
int old = atomicOr (int *addr, int val);      // *addr |= val
int old = atomicXor(int *addr, int val);      // *addr ^= val
int old = atomicExch(int *addr, int val);     // *addr = val (교환)
int old = atomicInc(unsigned *addr, unsigned wrap);  // *addr = ((*addr >= wrap) ? 0 : *addr + 1)

// 부동소수점 원자적 연산 (CUDA 2.0+)
float old = atomicAdd(float *addr, float val);  // *addr += val (FP32)
// 또한 사용 가능: double, half (Volta+)에 대한 atomicAdd

// 비교 후 교환: 잠금 없는 프로그래밍의 기반
int old = atomicCAS(int *addr, int compare, int val);
// 의미: if (*addr == compare) { *addr = val; } return 이전 *addr;
```

---

## 3. 비교 후 교환 (CAS): 범용 기본 연산

`atomicCAS`는 어떤 원자적 연산이든 구현할 수 있습니다:

```c
// 원자적 부동소수점 최댓값 (SM 8.0 이전에는 기본 제공 없음)
__device__ void atomicMaxFloat(float *addr, float val) {
    int *addr_as_int = (int *)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float current = __int_as_float(old);
        if (val <= current) return;  // 갱신 불필요
        int new_val = __float_as_int(val);
        old = atomicCAS(addr_as_int, expected, new_val);
    } while (old != expected);  // 다른 스레드가 addr를 바꾼 경우 재시도
}
```

CAS 루프 패턴:
1. 현재 값 읽기
2. 원하는 새 값 계산
3. CAS: 1단계와 3단계 사이에 아무도 변경하지 않은 경우만 갱신
4. CAS 실패 시 (다른 스레드가 변경했다면) 최신 값으로 재시도

이것이 모든 잠금 없는 자료 구조의 기반입니다.

---

## 4. 히스토그램 커널

히스토그램은 전형적인 원자적 연산 사용 사례입니다: 값이 빈(bin)에 나타나는 횟수 세기.

### 버전 1: 단순한 전역 원자적 연산

```c
__global__ void histogram_naive(const unsigned char *data, int *hist,
                                int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = (int)(data[i] * nbins / 256);
        atomicAdd(&hist[bin], 1);  // 모든 스레드가 ~256 빈에서 충돌
    }
}
```

**문제**: 높은 충돌 — 많은 스레드가 같은 빈에 동시에 쓰면 원자적 연산이 직렬화됩니다.

---

### 버전 2: 사유화된 히스토그램 (공유 메모리 + 원자적 병합)

각 블록이 공유 메모리에 자체 사유화된 히스토그램을 만든 다음 전역에 병합합니다:

```c
__global__ void histogram_privatized(const unsigned char *data, int *hist,
                                     int n, int nbins) {
    extern __shared__ int local_hist[];  // nbins 개의 int, 동적 할당

    // 공유 히스토그램 초기화
    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();

    // 공유 메모리에 누적 (낮은 충돌 — 블록 로컬)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = (int)(data[i] * nbins / 256);
        atomicAdd(&local_hist[bin], 1);  // 훨씬 낮은 충돌
    }
    __syncthreads();

    // 로컬 히스토그램을 전역에 병합 (블록당 빈당 하나의 병합)
    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        atomicAdd(&hist[i], local_hist[i]);
}

// 실행:
int sharedBytes = nbins * sizeof(int);
histogram_privatized<<<grid, block, sharedBytes>>>(d_data, d_hist, n, nbins);
```

**충돌 감소**: 전역 메모리에서 N번의 원자적 연산 대신, (N번의 블록 로컬 원자적 연산) + (gridSize × nbins번의 전역 병합). 대형 N의 경우 전역 원자적 충돌이 극적으로 적습니다.

---

## 5. 처리량 vs 충돌

**충돌 없음** (모든 스레드가 서로 다른 주소에 쓰기):

```c
// 각 스레드가 자신만의 위치에 씀 — 충돌 없음
atomicAdd(&hist[threadIdx.x], 1);  // 256개의 고유 주소

// 처리량: SM당 사이클당 ~1 원자적 연산 → 최대 처리량
```

**높은 충돌** (모든 스레드가 같은 주소에 씀):

```c
// 블록의 1024개 스레드 모두가 hist[0]에 씀
atomicAdd(&hist[0], 1);

// SM이 1024번의 쓰기를 직렬화 → 이 연산에만 1024 사이클
// 처리량: 충돌 없음 경우보다 ~1024배 느림
```

**벤치마크**:

```c
// 측정: 원자적 처리량이 충돌에 따라 어떻게 확장되는가?
// 설정: N=1000만 원소, 고유 빈 수 변화

// 빈 1개  (최대 충돌): ~ 45,000 μs
// 빈 4개:             ~ 11,500 μs
// 빈 64개:            ~  1,200 μs
// 빈 1024개 (낮음):   ~    320 μs
// 원자적 없음:        ~     80 μs

// 교훈: 사유화된 64빈이 사유화 없는 1024빈보다 낫다
```

---

## 6. Warp 집계 원자적 연산

전역 메모리를 공격하기 전에 warp 내에서 atomicAdd를 집계합니다:

```c
__device__ void warp_aggregated_add(int *addr, int val) {
    unsigned mask = __activemask();                    // 어느 스레드가 활성인가
    int leader    = __ffs(mask) - 1;                   // 가장 낮은 활성 레인

    // 동일한 주소에 쓰려는 스레드의 수를 셈
    // (여러 스레드가 같은 빈에 매핑될 때 히스토그램에 사용)
    unsigned match = __match_any_sync(mask, (unsigned long long)addr);
    int count      = __popc(match);                    // 같은 빈을 공격하는 스레드

    // 리더만 실제 원자적 연산 수행
    if ((mask & match) == match) {  // 나는 이 그룹의 리더
        int group_val = val * count;
        atomicAdd(addr, group_val);
    }
}
```

이는 많은 스레드가 같은 빈에 들어갈 때 원자적 연산을 32배까지 줄입니다 — 불균일 분포에서 결정적입니다.

---

## 7. 원자적 범위 (CUDA 9+)

원자적 연산은 SM, GPU, 또는 시스템 (NVLink/PCIe)으로 범위를 제한할 수 있습니다:

```c
#include <cuda/atomic>  // C++ 원자적 헤더 (libcu++)

// 블록 범위 원자적 (가장 빠름 — 블록 내에서만 가시적)
cuda::atomic<int, cuda::thread_scope_block> block_counter;

// 장치 범위 (기본값 — 이 GPU의 모든 스레드에게 가시적)
cuda::atomic<int, cuda::thread_scope_device> global_counter;

// 시스템 범위 (CPU와 모든 GPU에 가시적 — 가장 느림)
cuda::atomic<int, cuda::thread_scope_system> system_counter;
```

의미적으로 올바른 가장 좁은 범위를 사용하세요.

---

## 8. 완전한 예시: 병렬 문자 빈도 카운터

```c
#define ALPHA_SIZE 26

__global__ void letter_frequency(const char *text, int *freq, int n) {
    __shared__ int local_freq[ALPHA_SIZE];

    // 초기화
    if (threadIdx.x < ALPHA_SIZE) local_freq[threadIdx.x] = 0;
    __syncthreads();

    // 세기 (임의 N을 위한 스트라이드 루프)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < n;
             i += gridDim.x * blockDim.x) {
        char c = text[i];
        if (c >= 'a' && c <= 'z')
            atomicAdd(&local_freq[c - 'a'], 1);
        else if (c >= 'A' && c <= 'Z')
            atomicAdd(&local_freq[c - 'A'], 1);
    }
    __syncthreads();

    // 병합
    if (threadIdx.x < ALPHA_SIZE)
        atomicAdd(&freq[threadIdx.x], local_freq[threadIdx.x]);
}
```

---

## 핵심 요약

- `atomicAdd` / `atomicCAS`는 명시적 잠금 없이 병렬에서 올바른 읽기-수정-쓰기를 보장
- **충돌이 적**입니다: 많은 스레드가 같은 주소에 쓰면 직렬화 — 사유화 사용
- **사유화된 히스토그램**: 각 블록이 공유 메모리에 로컬 히스토그램을 만들고 병합; 전역 원자적 연산을 N에서 N_블록 × N_빈으로 감소
- `atomicCAS`는 범용 구성 요소 — 어떤 맞춤형 원자적 연산이든 구현 가능
- Warp 집계 원자적 연산은 병합 전에 같은 빈 쓰기를 결합하여 충돌을 32배까지 감소

---

**다음**: [08. 메모리 합치기](./08_Memory_Coalescing.md) — 128바이트 트랜잭션 세분성이 스트라이드 접근을 비싸게 만드는 방법, AoS vs SoA 레이아웃 비교, Nsight Compute로 스트라이드 패널티 측정.
