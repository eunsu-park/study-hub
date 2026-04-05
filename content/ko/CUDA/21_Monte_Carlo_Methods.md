# 21. GPU에서의 Monte Carlo Methods

**이전**: [N-Body Simulation](./20_N_Body_Simulation.md) | **다음**: [FFT on GPU](./22_FFT_on_GPU.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. cuRAND device API를 사용하여 GPU kernel 내부에서 난수 생성하기
2. 다양한 사용 사례에 맞는 XORWOW, MT19937, Sobol 생성기 선택하기
3. GPU 병렬 Monte Carlo π 추정기 구현하기
4. Monte Carlo 시뮬레이션으로 Black-Scholes 유럽형 콜 옵션 가격 계산하기
5. 추가 계산 없이 Monte Carlo 분산을 줄이는 대립 변수 기법 적용하기

---

## 1. GPU가 Monte Carlo에 뛰어난 이유

Monte Carlo 방법은 많은 독립 랜덤 샘플을 생성하고 각 샘플을 동일하게 처리합니다. 이는 SIMD GPU 모델에 완벽하게 적합합니다:

```
순차 Monte Carlo: 1 thread × N 샘플 × T 작업/샘플 = N×T
병렬 Monte Carlo: N thread × thread당 1 샘플 × T 작업/thread = T  (N배 속도향상)

N = 1000만 샘플: GPU가 1000만 샘플을 동시에 생성하고 처리
vs CPU: 1000만 개의 순차 샘플
```

주요 과제는 **난수 생성(RNG)**입니다: 각 thread는 독립적인 고품질 랜덤 스트림이 필요합니다.

---

## 2. cuRAND 개요

cuRAND는 두 가지 API를 제공합니다:

**Host API**: CPU에서 숫자를 생성하여 GPU로 전송 (미리 계산된 테이블에 유용):
```c
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

float *d_random;
cudaMalloc(&d_random, N * sizeof(float));
curandGenerateUniform(gen, d_random, N);  // 균일 [0,1)
curandDestroyGenerator(gen);
```

**Device API**: 각 thread가 자체 RNG 상태를 소유 — kernel 내부 난수 생성에 필요:
```c
#include <curand_kernel.h>

// thread당 하나의 RNG 상태 초기화
__global__ void init_rng(curandState *states, unsigned long long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        // 각 thread는 고유한 시퀀스를 가짐: 같은 seed, 다른 시퀀스 오프셋
        curand_init(seed, idx, 0, &states[idx]);
}

// kernel 내에서 상태 사용
__global__ void sample_kernel(curandState *states, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState local = states[idx];       // register에 복사 (더 빠름)
    float u = curand_uniform(&local);      // 균일 [0,1)
    float n = curand_normal(&local);       // 표준 정규 N(0,1)
    out[idx] = n;
    states[idx] = local;                   // 업데이트된 상태 기록
}
```

---

## 3. RNG 생성기 종류

```
생성기            주기              품질          비용      사용 사례
---------------------------------------------------------------------------
XORWOW (기본)     2^190 - 2^62     양호           낮음      범용
MT19937           2^19937 - 1      매우 양호       중간      고품질 균일
MRG32k3a          ~2^191           양호            중간      멀티-스트림 보장
Sobol (준난수)    N/A (결정론적)   낮은 불일치     중간      적분, 금융
Philox (카운터)   2^128            양호            매우 낮음 재현 가능, 인라인

실용적 선택:
  기본 시뮬레이션: CURAND_RNG_PSEUDO_XORWOW
  금융 Monte Carlo: CURAND_RNG_QUASI_SOBOL32 (더 빠른 수렴)
  재현 가능한 결과: Philox4_32_10 (카운터 기반, 저장할 상태 없음)
```

---

## 4. Monte Carlo로 π 추정

고전적 예시: [0,1]²에서 균일한 랜덤 (x, y)를 샘플링하고, 사분원 내의 점을 집계하여 π ≈ 4 × (내부 집계) / (총 샘플)을 추정합니다:

```c
#include <curand_kernel.h>

__global__ void estimate_pi(
    curandState *states, int *d_count, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState local = states[idx];

    int inside = 0;
    // 각 thread가 SAMPLES_PER_THREAD개 샘플을 추출
    const int SPT = 100;
    for (int k = 0; k < SPT; k++) {
        float x = curand_uniform(&local);
        float y = curand_uniform(&local);
        if (x*x + y*y <= 1.0f) inside++;
    }

    states[idx] = local;
    atomicAdd(d_count, inside);
}

double gpu_pi_estimate(int N_threads) {
    const int BLOCK = 256;
    int grid = (N_threads + BLOCK - 1) / BLOCK;

    curandState *d_states;
    cudaMalloc(&d_states, N_threads * sizeof(curandState));
    init_rng<<<grid, BLOCK>>>(d_states, 42ULL, N_threads);

    int *d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    estimate_pi<<<grid, BLOCK>>>(d_states, d_count, N_threads);

    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    long long total = (long long)N_threads * 100;  // thread당 100 샘플
    double pi = 4.0 * h_count / total;

    cudaFree(d_states); cudaFree(d_count);
    return pi;
}
// N_threads=100,000 (총 1000만 샘플)일 때: π ≈ 3.1416 (오차 ~0.01%)
```

---

## 5. Monte Carlo를 통한 Black-Scholes 옵션 가격 산정

**유럽형 콜 옵션**은 만기 T에서 max(S_T - K, 0)을 지급하는데, 여기서 S_T는 주식의 만기 가격입니다. 기하 브라운 운동에 의한 위험 중립 가격 결정:

```
S_T = S_0 · exp((r - 0.5·σ²)·T + σ·√T·Z)    Z ~ N(0,1)
콜 가격 C = e^(-r·T) · E[max(S_T - K, 0)]
```

```c
__global__ void black_scholes_mc(
    curandState *states,
    float  S0,    // 초기 주가
    float  K,     // 행사 가격
    float  r,     // 무위험 이자율 (연간)
    float  sigma, // 변동성 (연간)
    float  T,     // 만기까지 시간 (년)
    float *payoffs, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState local = states[idx];

    // 드리프트와 확산 사전 계산 (샘플당 register 연산 절약)
    float drift   = (r - 0.5f * sigma * sigma) * T;
    float diffuse = sigma * sqrtf(T);
    float disc    = expf(-r * T);

    // Box-Muller을 사용하여 표준 정규 추출 (curand_normal은 Marsaglia 사용)
    float Z  = curand_normal(&local);
    float ST = S0 * expf(drift + diffuse * Z);
    payoffs[idx] = disc * fmaxf(ST - K, 0.0f);

    states[idx] = local;
}

float option_price_mc(float S0, float K, float r, float sigma, float T, int N) {
    const int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    curandState *d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));
    init_rng<<<grid, BLOCK>>>(d_states, 12345ULL, N);

    float *d_payoffs;
    cudaMalloc(&d_payoffs, N * sizeof(float));
    black_scholes_mc<<<grid, BLOCK>>>(d_states, S0, K, r, sigma, T, d_payoffs, N);

    // payoffs를 평균으로 reduce (CUB DeviceReduce 또는 thrust::reduce 사용)
    float total = thrust_reduce_sum(d_payoffs, N);
    float price = total / N;

    cudaFree(d_states); cudaFree(d_payoffs);
    return price;
}
// 폐쇄형 Black-Scholes: S0=100, K=100, r=0.05, σ=0.2, T=1일 때 ~$10.45
// Monte Carlo (N=1000만): 오차 < 0.01 달러
```

---

## 6. 대립 변수 (분산 감소)

모든 샘플 Z ~ N(0,1)에 대해 -Z도 평가합니다. 큰 양수 Z는 높은 수익을 주고 -Z는 낮은 수익을 주므로, 두 평균은 어느 하나만의 분산보다 낮습니다:

```c
__global__ void black_scholes_antithetic(
    curandState *states,
    float S0, float K, float r, float sigma, float T,
    float *payoffs, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N / 2) return;  // N/2 thread, 각각 2개 샘플 생성

    curandState local = states[idx];
    float drift   = (r - 0.5f * sigma * sigma) * T;
    float diffuse = sigma * sqrtf(T);
    float disc    = expf(-r * T);

    float Z  = curand_normal(&local);

    // 원본 경로 (+Z)
    float ST1 = S0 * expf(drift + diffuse * Z);
    float p1  = disc * fmaxf(ST1 - K, 0.0f);

    // 대립 경로 (-Z)
    float ST2 = S0 * expf(drift - diffuse * Z);
    float p2  = disc * fmaxf(ST2 - K, 0.0f);

    // 쌍의 평균 저장
    payoffs[idx] = 0.5f * (p1 + p2);
    states[idx]  = local;
}
// 대립 변수 사용 시: 동일한 N 샘플 → ~50% 분산 감소
// 동등하게: ~50% 더 적은 샘플로 동일한 정확도 달성
```

**작동 원리**: 볼록 수익에 대해 Z와 -Z는 음의 상관관계를 가지므로, 평균을 내면 랜덤 변동의 많은 부분이 상쇄됩니다. 콜 옵션 (볼록 수익)에 대해 대립 변수는 일반적으로 표준 오차를 40–70% 줄입니다.

---

## 7. Sobol 준-난수 수열

의사 랜덤 숫자는 군집화될 수 있습니다; Sobol 수열 (낮은 불일치)은 공간을 더 균일하게 채워 적분에서 더 나은 수렴을 제공합니다:

```c
// cuRAND로 Sobol 수열 생성
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32);
curandSetQuasiRandomGeneratorDimensions(gen, 2);  // (x, y) 쌍을 위한 2D Sobol

float *d_sobol;
cudaMalloc(&d_sobol, 2 * N * sizeof(float));
curandGenerateUniform(gen, d_sobol, 2 * N);
// d_sobol[0..N-1] = x 좌표, d_sobol[N..2N-1] = y 좌표

// 스크램블된 Sobol (더 나은 통계적 특성)
curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
```

**N 샘플 대비 적분 오차 수렴 비교:**
```
방법              오차율            N=1000만에서의 오차
---------------------------------------------------
의사-난수 MC      O(1/√N)          ~0.0003
Sobol 준-MC       O((log N)^d / N) ~0.00001  (5-10배 더 좋음)
```

Sobol 수열은 금융 Monte Carlo의 업계 표준입니다.

---

## 핵심 요약

- cuRAND **device API** (`curandState`, `curand_init`, `curand_uniform`)는 각 thread에 독립적인 RNG 스트림을 제공합니다
- 사용 전에 `curandState`를 항상 로컬 register 변수에 복사하세요 — 추출 시 전역 메모리 왕복을 반복하지 않습니다
- **XORWOW**이 기본값; **Sobol**은 적분에 낮은 불일치 수열을 제공; **Philox**는 카운터 기반으로 초기화 비용이 가장 낮습니다
- GPU Monte Carlo는 선형으로 확장됩니다: 1000만 개의 독립 샘플이 1개 샘플과 같은 시간에 실행되며, 오직 계산 처리량에만 제한됩니다
- **대립 변수**는 추가 메모리 비용 없이 분산을 절반으로 줄입니다 (또는 필요한 샘플 수를 절반으로) — 콜/풋 옵션에는 항상 활성화할 가치가 있습니다
- GPU에서 **Black-Scholes MC** (N=1000만 샘플)는 ~10ms에 실행됩니다; 정확한 폐쇄형은 < 1μs이지만 MC는 이색 수익으로 일반화됩니다

---

**다음**: [22. FFT on GPU](./22_FFT_on_GPU.md) — cuFFT로 1D/2D/3D Fast Fourier Transform을 계산하고, FFT를 통한 합성곱을 구현하며, 정규화 함정을 이해합니다.
