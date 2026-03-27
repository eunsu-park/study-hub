# 04. 최적화된 행렬 곱셈

**이전**: [텐서 연산과 BLAS](./03_Tensor_Ops_BLAS.md) | **다음**: [Autograd 엔진](./05_Autograd_Engine.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Naive matmul이 캐시 비효율적인 이유와 미스 패널티 정량화 설명
2. L1/L2 캐시에 데이터를 유지하는 루프 타일 SGEMM 구현
3. 연산 밀도를 위한 레지스터 수준 마이크로 커널 블로킹 추가
4. AVX2 intrinsics로 명령당 8개의 float 처리
5. `perf stat`으로 프로파일링하고 달성된 GFLOP/s 측정

---

## 1. Naive Matmul이 느린 이유

Naive 3중 루프를 회상해 봅시다:

```c
for (i) for (j) for (k)
    C[i][j] += A[i][k] * B[k][j];
```

`N=1024`에서 `B[k][j]` 접근 패턴:
- `k`가 증가 → stride `N=1024`개 원소 = **4096바이트**
- `B[k][j]` 접근마다 다른 캐시 라인 → **매 반복마다 캐시 미스**
- L1 캐시는 `32KB / 4 = 8192`개의 float를 저장할 수 있지만 열 방향 접근으로 모두 날아감

**결과**: 시간의 ~99%가 메모리 대기에 소비됨, 계산 아님.

---

## 2. 루프 타일링 (블로킹)

**타일링**은 L1/L2 캐시에 맞는 작은 블록(타일)으로 행렬을 처리합니다.

```c
// 타일 크기 TILE의 타일드 matmul
#define TILE 64   // 64 floats = 256바이트 = 캐시 라인 4개 — L1에 맞음

void matmul_tiled(float *C, const float *A, const float *B,
                  size_t M, size_t K, size_t N) {
    memset(C, 0, M * N * sizeof(float));

    for (size_t i0 = 0; i0 < M; i0 += TILE)
    for (size_t j0 = 0; j0 < N; j0 += TILE)
    for (size_t k0 = 0; k0 < K; k0 += TILE) {
        size_t imax = i0 + TILE < M ? i0 + TILE : M;
        size_t jmax = j0 + TILE < N ? j0 + TILE : N;
        size_t kmax = k0 + TILE < K ? k0 + TILE : K;

        // 타일 처리 — 캐시에 유지됨
        for (size_t i = i0; i < imax; i++)
        for (size_t k = k0; k < kmax; k++) {
            float a_ik = A[i * K + k];      // 한 번 로드, j에 걸쳐 재사용
            for (size_t j = j0; j < jmax; j++)
                C[i * N + j] += a_ik * B[k * N + j];
        }
    }
}
```

**작동 원리**:
- 타일 `A[i0..imax, k0..kmax]`가 L1에 맞음: `TILE*TILE*4 = 16KB` (TILE=64의 경우)
- 타일 `B[k0..kmax, j0..jmax]`도 L1에 맞음
- 내부 루프는 `B`에 연속적으로 접근 → 캐시 친화적

**성능 향상**: 큰 N에서 naive 대비 일반적으로 **4–10× 빠름**.

---

## 3. 레지스터 블로킹 (마이크로 커널)

타일링 후, **레지스터만** 사용하여 작은 출력 블록을 계산함으로써 더 개선할 수 있습니다.

`4×4` 마이크로 커널은 `4×4 = 16`개의 C 값을 완전히 CPU 레지스터에서 누적합니다:

```c
// 4×4 레지스터 마이크로 커널
// C[i:i+4, j:j+4] += A[i:i+4, k] * B[k, j:j+4] 계산
static inline void micro_kernel_4x4(
    float *C, const float *A, const float *B,
    size_t K, size_t N, size_t i, size_t j)
{
    // 레지스터에 4×4 출력 누산기 로드
    float c00=0, c01=0, c02=0, c03=0;
    float c10=0, c11=0, c12=0, c13=0;
    float c20=0, c21=0, c22=0, c23=0;
    float c30=0, c31=0, c32=0, c33=0;

    for (size_t k = 0; k < K; k++) {
        float a0 = A[(i+0)*K + k],  a1 = A[(i+1)*K + k];
        float a2 = A[(i+2)*K + k],  a3 = A[(i+3)*K + k];
        float b0 = B[k*N + (j+0)],  b1 = B[k*N + (j+1)];
        float b2 = B[k*N + (j+2)],  b3 = B[k*N + (j+3)];

        c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
        c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
        c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
        c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
    }

    // C에 다시 쓰기
    C[(i+0)*N+j+0]+=c00; C[(i+0)*N+j+1]+=c01;
    /* ... 나머지 모두 ... */
}
```

컴파일러는 `c00..c33`을 레지스터에 유지하여 모든 중간 메모리 쓰기를 제거합니다.

---

## 4. AVX2 Intrinsics

**AVX2**(Advanced Vector Extensions 2)는 256비트 레지스터(`__m256`)를 사용하여 **단일 명령으로 8개의 float**를 처리합니다.

```
스칼라 FMA:  multiply-add 1개 → 사이클당 2 FLOP
AVX2   FMA:  multiply-add 8개 → 사이클당 16 FLOP   (8× 향상)
```

### 주요 Intrinsics

```c
#include <immintrin.h>

__m256 a = _mm256_loadu_ps(ptr);       // 비정렬 8개 float 로드
__m256 b = _mm256_set1_ps(scalar);    // 스칼라를 8개 float로 브로드캐스트
__m256 c = _mm256_fmadd_ps(a, b, c);  // c = a*b + c (FMA)
_mm256_storeu_ps(ptr, c);             // 8개 float 저장
```

### AVX2 Matmul 내부 루프

```c
#include <immintrin.h>

// AVX2로 C의 한 행 계산: C[i, 0..N] += A[i,k] * B[k, 0..N]
void matmul_row_avx2(float *C_row, const float *B_row,
                     float a_scalar, size_t N) {
    __m256 a_vec = _mm256_set1_ps(a_scalar);  // A[i,k] 브로드캐스트

    size_t j = 0;
    for (; j + 8 <= N; j += 8) {
        __m256 b_vec = _mm256_loadu_ps(B_row + j);
        __m256 c_vec = _mm256_loadu_ps(C_row + j);
        c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);  // c += a * b
        _mm256_storeu_ps(C_row + j, c_vec);
    }
    // 나머지 처리 (N이 8의 배수가 아닌 경우)
    for (; j < N; j++)
        C_row[j] += a_scalar * B_row[j];
}
```

**컴파일**: `gcc -std=c11 -O3 -march=native -mavx2 -mfma`

> **AVX2 지원 확인**: `grep avx2 /proc/cpuinfo` (Linux) 또는 `sysctl -a | grep avx2` (macOS)

---

## 5. 타일링 + AVX2 결합

최고의 단일 스레드 matmul은 두 가지를 결합합니다:

```c
#define TILE_M 64
#define TILE_N 64
#define TILE_K 256

void matmul_tiled_avx2(float *C, const float *A, const float *B,
                        size_t M, size_t K, size_t N) {
    memset(C, 0, M * N * sizeof(float));

    for (size_t i0 = 0; i0 < M; i0 += TILE_M)
    for (size_t j0 = 0; j0 < N; j0 += TILE_N)
    for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
        /* 경계 계산 ... */
        for (size_t i = i0; i < imax; i++) {
            const float *A_row = A + i * K + k0;
            float       *C_row = C + i * N + j0;
            size_t       jlen  = jmax - j0;

            for (size_t k = 0; k < klen; k++) {
                float a_val = A_row[k];
                const float *B_row = B + (k0 + k) * N + j0;
                matmul_row_avx2(C_row, B_row, a_val, jlen);
            }
        }
    }
}
```

---

## 6. 성능 비교

| N | Naive | Tiled | Tiled+AVX2 | OpenBLAS |
|---|-------|-------|------------|----------|
| 256 | 20 ms | 3 ms | 0.8 ms | 0.2 ms |
| 512 | 160 ms | 18 ms | 5 ms | 1.3 ms |
| 1024 | ~1400 ms | 130 ms | 35 ms | 11 ms |

**N=1024 Roofline 분석**:
- 총 FLOPs: `2 * 1024^3 ≈ 2.1 GFLOP`
- Naive: `1400 ms → 1.5 GFLOP/s` (피크의 2%)
- Tiled+AVX2: `35 ms → 60 GFLOP/s` (단일 코어 피크의 80%)
- OpenBLAS: `11 ms → 190 GFLOP/s` (멀티스레드)

---

## 7. `perf`로 프로파일링

Linux에서 캐시 동작을 프로파일링합니다:

```bash
# 캐시 미스 비교: naive vs tiled
perf stat -e cache-misses,cache-references,instructions,cycles ./benchmark 1024 naive
perf stat -e cache-misses,cache-references,instructions,cycles ./benchmark 1024 tiled
```

**예상 출력 (naive, N=1024)**:
```
  12,589,123,456      cache-misses              # 캐시 참조의 98.2%
```

**예상 출력 (tiled, N=1024)**:
```
     45,123,456      cache-misses              # 캐시 참조의 3.5%
```

타일 버전은 **280× 적은 캐시 미스**를 가집니다 — 여기서 속도 향상이 나옵니다.

---

## 핵심 요약

- Naive matmul의 `B[k][j]` 열 접근은 L1 캐시 스래싱을 유발합니다 — 지배적인 병목
- **루프 타일링**: 활성 데이터를 L1/L2 캐시에 유지; 타일 크기는 `3 * TILE^2 * 4 ≤ L1_size`를 만족해야 함
- **레지스터 블로킹** (4×4 마이크로 커널): 메모리 로드당 연산을 최대화
- **AVX2 FMA**: 사이클당 8개의 multiply-add — 스칼라 처리량의 8×
- 타일링 + AVX2 결합으로 단일 코어 피크 FLOP/s의 ~80% 달성

---

**다음**: [05. Autograd 엔진](./05_Autograd_Engine.md) — 함수 포인터와 위상 정렬을 사용하여 C로 연산 그래프와 자동 미분 엔진을 구축합니다.
