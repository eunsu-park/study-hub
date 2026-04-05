# 27. 난수 및 확률적 방법 (Random Numbers and Stochastic Methods)

**이전**: [Image Processing GPU](./26_Image_Processing_GPU.md) | **다음**: [Thrust and CUB](./28_Thrust_and_CUB.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. cuRAND host API를 사용하여 대규모 난수 배치를 효율적으로 생성하기
2. kernel 내부에서 thread당 랜덤 스트림을 위해 cuRAND device API 사용하기
3. GPU에서 병렬 Monte Carlo 적분 및 Metropolis-Hastings MCMC 구현하기
4. 체커보드 Metropolis 업데이트를 사용한 2D Ising 모델 시뮬레이션하기
5. 통계 오차를 줄이기 위한 분산 감소 기법 (대립 변수, 중요도 샘플링) 적용하기

---

## 1. cuRAND 개요

cuRAND는 GPU 가속 난수 생성을 제공합니다. 두 가지 사용 모드를 지원합니다:

```
Host API:
  curandCreateGenerator() → device 메모리에 직접 생성
  장점: 간단, 모든 상태 관리 처리
  단점: 전체 배치가 한 번에 생성됨 (큰 메모리 사용량)

Device API:
  curand_init()이 thread당 상태 초기화
  curand_uniform()이 kernel 내부에서 호출당 하나의 숫자 생성
  장점: 즉석 생성, 추가 저장소 불필요
  단점: 상태 초기화 비용 (~100 사이클); thread당 상태 48-192바이트
```

**지원 생성기:**

| 생성기 | 주기 | 품질 | 속도 |
|--------|------|------|------|
| XORWOW    | ~2^190 | 양호    | 가장 빠름 |
| Philox4   | 2^128  | 높음    | 빠름 |
| MRG32k3a  | ~2^191 | 높음    | 보통 |
| MTGP32    | 2^11213| 매우 높음 | 보통 (shared memory 기반) |
| Sobol32   | 준난수  | QMC     | 빠름 (낮은 불일치) |

---

## 2. Host API: 배치 생성

```c
#include <curand.h>

void generate_uniform_host(float **d_rand, int N) {
    curandGenerator_t gen;

    // 생성기 생성
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);

    // seed 설정 (재현성을 위해)
    curandSetPseudoRandomGeneratorSeed(gen, 12345ULL);

    // device 메모리 할당
    cudaMalloc(d_rand, N * sizeof(float));

    // [0, 1)에서 N개의 균일 float 생성
    curandGenerateUniform(gen, *d_rand, N);

    // 정규 분포 N(0, 1) 생성
    float *d_normal;
    cudaMalloc(&d_normal, N * sizeof(float));
    curandGenerateNormal(gen, d_normal, N, 0.0f, 1.0f);  // mean=0, std=1

    // Box-Muller은 짝수 N 필요; 로그 정규의 경우:
    // curandGenerateLogNormal(gen, d_log, N, mean, std);

    curandDestroyGenerator(gen);
    cudaFree(d_normal);
}

// 준-난수 Sobol 수열 (낮은 불일치 — 의사 난수보다 더 나은 수렴)
void generate_sobol(float **d_sobol, int N, int dims) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(gen, dims);
    cudaMalloc(d_sobol, N * dims * sizeof(float));
    curandGenerateUniform(gen, *d_sobol, N * dims);
    curandDestroyGenerator(gen);
}
```

---

## 3. Device API: Thread당 생성

```c
#include <curand_kernel.h>

// thread당 하나의 curand 상태 초기화
// 한 번 호출; kernel 실행 간 재사용을 위해 전역 메모리에 상태 저장
__global__ void init_rng(curandState *states, int N, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // sequence = tid로 각 thread가 독립적인 스트림을 가짐
    curand_init(seed, /*sequence=*/tid, /*offset=*/0, &states[tid]);
}

// Monte Carlo π 추정: 단위 원 내의 점 집계
__global__ void monte_carlo_pi(curandState *states, int *counts, int samples_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = states[tid];  // register에 로드

    int inside = 0;
    for (int s = 0; s < samples_per_thread; s++) {
        float x = curand_uniform(&local_state);   // [0, 1)
        float y = curand_uniform(&local_state);
        if (x*x + y*y <= 1.0f) inside++;
    }

    states[tid] = local_state;  // 상태 저장 (중요!)
    counts[tid] = inside;
}

// 호스트: π 추정
void estimate_pi(int total_samples) {
    const int THREADS = 256, BLOCKS = 1024;
    const int N = THREADS * BLOCKS;
    const int spt = total_samples / N;  // thread당 샘플 수

    curandState *d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));
    init_rng<<<BLOCKS, THREADS>>>(d_states, N, 42ULL);

    int *d_counts;
    cudaMalloc(&d_counts, N * sizeof(int));
    monte_carlo_pi<<<BLOCKS, THREADS>>>(d_states, d_counts, spt);

    // 집계 reduce
    int total_inside = thrust_reduce_sum(d_counts, N);
    double pi = 4.0 * total_inside / (double)(N * spt);
    printf("pi ≈ %.6f (오차: %.2e)\n", pi, fabs(pi - M_PI));
}
```

---

## 4. Metropolis-Hastings MCMC

Metropolis-Hastings는 정규화 상수를 알지 못해도 목표 분포 π(x)에서 샘플링합니다:

```
단계당 알고리즘:
  1. x' = x + ε를 제안,   ε ~ N(0, σ²)
  2. 수용 비율 α = min(1, π(x') / π(x))
  3. 확률 α로 x' 수용; 아니면 x에 머뭄

병렬 MCMC: M개의 독립적인 체인을 동시에 실행 (완전 병렬)
```

```c
// 목표: 2D 상관 가우시안
// π(x,y) ∝ exp(-0.5 * [x,y] Σ^{-1} [x,y]^T)
__device__ float log_target(float x, float y) {
    // Σ = [[1, 0.9],[0.9, 1]]  → Σ^{-1} = [[1,-0.9],[-0.9,1]] / (1-0.81)
    float det_inv = 1.f / (1.f - 0.9f*0.9f);
    return -0.5f * det_inv * (x*x - 2.f*0.9f*x*y + y*y);
}

__global__ void metropolis_2d(
    curandState *states,
    float *chain_x, float *chain_y,   // 출력 체인 [N * n_steps]
    int N, int n_steps, float step_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandState local = states[tid];
    float x = curand_normal(&local);  // 랜덤 점에서 시작
    float y = curand_normal(&local);
    float log_p = log_target(x, y);

    for (int s = 0; s < n_steps; s++) {
        float xp = x + step_size * curand_normal(&local);
        float yp = y + step_size * curand_normal(&local);
        float log_pp = log_target(xp, yp);

        float log_alpha = log_pp - log_p;
        float u = curand_uniform(&local);
        if (logf(u) < log_alpha) {
            x = xp; y = yp; log_p = log_pp;
        }
        chain_x[tid * n_steps + s] = x;
        chain_y[tid * n_steps + s] = y;
    }
    states[tid] = local;
}
```

---

## 5. 2D Ising 모델 (병렬 Metropolis)

정사각형 격자의 Ising 모델은 체커보드 분해를 사용하여 읽기-쓰기 충돌 없이 Metropolis 업데이트를 병렬화합니다:

```
에너지: E = -J Σ_{<i,j>} s_i s_j   (s_i = ±1)
수용: P(flip) = min(1, exp(-ΔE / kT))

체커보드 (적-흑 순서):
  짝수 사이트 (i+j 짝수):  병렬로 업데이트 (이웃 충돌 없음)
  홀수 사이트 (i+j 홀수):  병렬로 업데이트
```

```c
// Ising 스핀은 int8로 저장: +1 또는 -1
__global__ void ising_sweep(
    int8_t *spins, int Nx, int Ny,
    float beta,      // β = 1/(k_B T)
    int parity,      // 0 = 짝수 사이트 업데이트, 1 = 홀수 사이트 업데이트
    curandState *states)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    if ((x + y) % 2 != parity) return;  // 잘못된 색상 건너뜀

    int tid = y * Nx + x;
    curandState local = states[tid];

    int8_t s = spins[tid];

    // 4개 이웃의 합 (주기적 BC)
    int sum_nbr =
        spins[((y-1+Ny)%Ny) * Nx + x] +
        spins[((y+1)%Ny)    * Nx + x] +
        spins[y * Nx + (x-1+Nx)%Nx]  +
        spins[y * Nx + (x+1)%Nx];

    // ΔE = 2 * J * s * sum_nbr  (J=1)
    float dE = 2.f * s * sum_nbr;

    // 뒤집기 수용/거부
    if (dE <= 0.f || curand_uniform(&local) < expf(-beta * dE))
        spins[tid] = -s;

    states[tid] = local;
}

// 스핀당 자기화 측정
float measure_magnetization(const int8_t *d_spins, int N) {
    // Thrust reduce 사용 (레슨 28 참조)
    // thrust::reduce(thrust::device_pointer_cast(d_spins), ...) / N
    return 0.f; // 자리 표시자
}
```

---

## 6. 분산 감소 기법

표준 Monte Carlo는 O(1/√N)으로 수렴합니다. 이 방법들은 더 많은 샘플 없이 분산을 줄입니다:

### 대립 변수

U ~ Uniform(0,1)인 함수 f(U)에 대해 쌍 (U, 1-U) 사용:

```c
__global__ void mc_antithetic(curandState *states, float *results, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N/2) return;

    curandState local = states[tid];
    float sum = 0.f;
    const int S = 100;

    for (int s = 0; s < S; s++) {
        float u = curand_uniform(&local);
        float u_anti = 1.f - u;
        // f(u) = exp(-u): 0에서 1까지 exp(-u) 적분 → 정확한 값 = 1 - 1/e ≈ 0.6321
        float f1 = expf(-u);
        float f2 = expf(-u_anti);
        sum += 0.5f * (f1 + f2);   // 쌍의 평균
    }
    results[tid] = sum / S;
    states[tid] = local;
}
// 대립 변수는 단조 함수에 대해 분산을 ~50% 감소 (추가 랜덤 호출 없음)
```

### 중요도 샘플링

|f(x)|에 근사하는 제안 분포 q(x)에서 샘플링하고, f(x)/q(x)로 가중:

```
추정기: (1/N) Σ f(x_i) / q(x_i),  x_i ~ q
최적 q(x) ∝ |f(x)|  → 제로 분산
```

```c
// 지수 기울기를 사용하여 X~N(0,1)에 대한 꼬리 확률 P(X > t) 추정
// 제안 q(x) = λ·exp(-λ(x-t)),  x > t
__global__ void importance_sampling_tail(
    curandState *states, float *results,
    int N, float t, float lambda)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    curandState local = states[tid];

    // t에서 시작하는 Exponential(lambda)에서 샘플링
    float x = t - logf(curand_uniform(&local)) / lambda;  // 역 CDF

    // 중요도 가중치: N(0,1) / Exponential(lambda)
    float log_p = -0.5f * x * x - 0.5f * logf(2.f * M_PI);   // log N(0,1)
    float log_q = logf(lambda) - lambda * (x - t);             // log Exp 제안
    float w = expf(log_p - log_q);                             // 우도 비율

    results[tid] = w;   // E[w] ≈ P(X > t)
    states[tid] = local;
}
```

---

## 7. 통계 수렴 테스트

```c
// 균일성 카이 제곱 테스트: histogram bin을 예상 집계와 비교
void chi_squared_test(const float *d_rand, int N, int bins) {
    // 1. histogram 계산 (atomic, 레슨 18 참조)
    int *h_hist = compute_histogram_cpu(d_rand, N, bins);

    float expected = (float)N / bins;
    float chi2 = 0.f;
    for (int b = 0; b < bins; b++) {
        float diff = h_hist[b] - expected;
        chi2 += diff * diff / expected;
    }

    // 자유도 = bins - 1
    // 유의수준 0.05, df=bins-1에서의 임계값
    float critical = chi2_critical(bins - 1, 0.05);
    printf("χ² = %.2f, 임계값 = %.2f → %s\n",
           chi2, critical,
           chi2 < critical ? "합격 (균일)" : "불합격 (비균일)");
    free(h_hist);
}
```

---

## 핵심 요약

- **cuRAND host API**는 단일 호출로 GPU에서 직접 배치를 생성합니다; 메모리가 충분할 때 사전 생성 샘플에 사용하세요
- **cuRAND device API**는 thread당 즉석에서 생성합니다; 스트림 독립성을 유지하기 위해 kernel 실행 간에 `curandState`를 전역 메모리에 저장/복원하세요
- **병렬 체인**: M개의 독립적인 체인을 가진 Metropolis-Hastings는 완전 병렬; 체커보드 순서는 Ising과 같은 격자 기반 모델의 단일 체인 공간 병렬성을 가능하게 합니다
- **2D Ising 모델**: 체커보드 (적-흑) 스윕은 동일한 색상 이웃만 상호작용하므로 스핀의 절반이 동시에 업데이트할 수 있습니다
- **대립 변수**는 단조 피적분함수에 대해 추가 RNG 비용 없이 분산을 ~50% 줄이기 위해 쌍 샘플 (U, 1-U)을 사용합니다
- **중요도 샘플링**은 기여가 높은 영역에 샘플을 집중시킵니다; 표준 MC가 실용적이지 않은 희귀 이벤트 추정에 가장 강력합니다

---

**다음**: [28. Thrust and CUB](./28_Thrust_and_CUB.md) — 커스텀 kernel을 작성하지 않고 고수준 GPU 정렬, reduction, scan을 위한 Thrust STL 유사 라이브러리와 CUB 기본 요소를 사용합니다.
