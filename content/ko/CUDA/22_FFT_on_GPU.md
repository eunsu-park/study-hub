# 22. GPU에서의 FFT

**이전**: [Monte Carlo Methods](./21_Monte_Carlo_Methods.md) | **다음**: [PDE Solvers Heat Equation](./23_PDE_Solvers_Heat_Equation.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. cuFFT API로 GPU에서 1D, 2D, 3D FFT 계산하기
2. 단일 호출로 많은 독립 변환을 위한 배치 FFT 실행하기
3. FFT를 통한 합성곱 (O(N log N)) 구현 및 직접 합성곱 (O(N·K))과 비교하기
4. 신호에서 주파수 성분을 제거하는 스펙트럼 필터링 적용하기
5. cuFFT의 정규화 규칙 이해 및 1/N 스케일 팩터 올바르게 적용하기

---

## 1. 이산 푸리에 변환 (DFT)

DFT는 N개의 복소수를 시간/공간 영역에서 주파수 영역으로 변환합니다:

```
X[k] = Σ_{n=0}^{N-1} x[n] · e^(-2πi·k·n/N)    k = 0, 1, ..., N-1

역 DFT:
x[n] = (1/N) · Σ_{k=0}^{N-1} X[k] · e^(2πi·k·n/N)

핵심 특성: DFT는 신호를 N개의 주파수 성분(사인파)으로 분해합니다.
FFT는 DFT를 O(N²) 대신 O(N log N)으로 계산합니다.
```

cuFFT는 Cooley-Tukey 기수-2 FFT와 그 일반화를 구현하며, GPU 실행에 최적화되어 있습니다.

---

## 2. cuFFT 기초: 1D FFT

```c
#include <cufft.h>

void fft_1d_example(int N) {
    // device에 복소수 배열 할당
    cufftComplex *d_signal;
    cudaMalloc(&d_signal, N * sizeof(cufftComplex));

    // 플랜 생성: N 복소수-복소수 포인트의 1D FFT
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, /*batch=*/1);

    // 순방향 FFT: CUFFT_FORWARD = -1 (지수 부호 규칙)
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
    // d_signal은 이제 X[k]를 포함 (정규화되지 않음)

    // 역 FFT: CUFFT_INVERSE = +1
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE);
    // d_signal은 이제 N * x[n]을 포함  ← cuFFT는 정규화하지 않습니다!

    // 역 FFT 후 반드시 N으로 나눠야 합니다:
    scale_kernel<<<(N + 255) / 256, 256>>>(d_signal, 1.0f / N, N);

    cufftDestroy(plan);
    cudaFree(d_signal);
}

// 정규화 kernel
__global__ void scale_complex(cufftComplex *data, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}
```

**정규화 규칙**: cuFFT는 정규화를 적용하지 않습니다 — 순방향 및 역 FFT 모두 비정규화 상태입니다. 순방향 + 역방향 후 각 요소는 N이 곱해집니다. `CUFFT_INVERSE` 후에는 항상 N으로 나누세요.

---

## 3. 실수-복소수 FFT (R2C)

실수 값 입력 (예: 오디오, 센서 데이터)에 대해 cuFFT는 켤레 대칭을 활용하여 N/2+1개의 복소수 출력 계수만 저장합니다:

```c
void fft_real_1d(int N) {
    float        *d_real;     // N개 실수 입력 값
    cufftComplex *d_freq;     // N/2 + 1개 복소수 출력 값 (에르미트)

    cudaMalloc(&d_real, N * sizeof(float));
    cudaMalloc(&d_freq, (N / 2 + 1) * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_R2C, 1);
    cufftExecR2C(plan, d_real, d_freq);
    // d_freq[0] = DC 성분 (실수)
    // d_freq[k] = 주파수 k / (N * dt)에서의 복소 진폭
    // d_freq[N/2] = Nyquist 주파수 (실수)

    // 역: C2R
    cufftHandle iplan;
    cufftPlan1d(&iplan, N, CUFFT_C2R, 1);
    cufftExecC2R(iplan, d_freq, d_real);
    scale_real_kernel<<<(N+255)/256, 256>>>(d_real, 1.0f/N, N);

    cufftDestroy(plan); cufftDestroy(iplan);
    cudaFree(d_real); cudaFree(d_freq);
}
```

R2C는 실수 입력에 대해 C2C보다 메모리를 ~50% 절약하고 ~50% 빠르게 실행됩니다.

---

## 4. 2D 및 3D FFT

```c
// 2D FFT (예: 이미지 처리, 2D 합성곱)
void fft_2d(int Nx, int Ny) {
    cufftComplex *d_image;
    cudaMalloc(&d_image, Nx * Ny * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan2d(&plan, Ny, Nx, CUFFT_C2C);  // 주의: 행 먼저 (Ny, Nx)
    cufftExecC2C(plan, d_image, d_image, CUFFT_FORWARD);
    // 각 요소 d_image[ky*Nx + kx] = X[ky, kx]

    cufftDestroy(plan);
    cudaFree(d_image);
}

// 3D FFT (예: 체적 데이터, 3D 합성곱)
void fft_3d(int Nx, int Ny, int Nz) {
    cufftComplex *d_vol;
    cudaMalloc(&d_vol, Nx * Ny * Nz * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan3d(&plan, Nz, Ny, Nx, CUFFT_C2C);
    cufftExecC2C(plan, d_vol, d_vol, CUFFT_FORWARD);

    cufftDestroy(plan);
    cudaFree(d_vol);
}
```

---

## 5. 배치 FFT

단일 cuFFT 호출로 동일한 크기의 많은 독립 FFT를 계산합니다 — 신경망, 오디오 처리, 다중 채널 신호 분석에 필수적입니다:

```c
// 배치 1D FFT: 크기 N의 B개 독립 FFT를 동시에 계산
void batched_fft_1d(int N, int B) {
    cufftComplex *d_signals;
    // 레이아웃: 배치 b는 d_signals[b * N ... b*N + N - 1]에 있음
    cudaMalloc(&d_signals, B * N * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, B);  // 세 번째 인자 = 배치 수
    cufftExecC2C(plan, d_signals, d_signals, CUFFT_FORWARD);

    cufftDestroy(plan);
    cudaFree(d_signals);
}

// 고급: 비연속 배치 또는 스트라이드 레이아웃을 위한 PlanMany
void batched_fft_strided(int N, int B, int stride) {
    cufftHandle plan;
    int rank  = 1;           // 1D 변환
    int n[]   = {N};
    int inembed[] = {0};     // NULL 동등 (자동)
    int onembed[] = {0};
    int idist = stride;      // 입력에서 배치 시작 간 거리
    int odist = stride;      // 출력에서 배치 시작 간 거리
    int istride = 1;
    int ostride = 1;

    cufftPlanMany(&plan, rank, n,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_C2C, B);
    // ... exec 및 destroy
}
```

---

## 6. FFT를 통한 합성곱

**직접 합성곱**: 길이 N 신호와 길이 K 커널에 대해 O(N·K).
**FFT 합성곱**: 합성곱 정리를 통해 O(N log N): conv(x, h) = IFFT(FFT(x) · FFT(h))

```c
// 1D 합성곱 via FFT
// x: 길이 N 신호, h: 길이 K 커널, 출력 길이 = N + K - 1
void fft_convolution(
    const float *d_x, int N,
    const float *d_h, int K,
    float *d_out)
{
    int M = N + K - 1;                // 선형 합성곱 길이
    int padded = next_power_of_two(M); // cuFFT는 2의 거듭제곱에서 가장 빠름

    cufftComplex *d_X, *d_H;
    cudaMalloc(&d_X, padded * sizeof(cufftComplex));
    cudaMalloc(&d_H, padded * sizeof(cufftComplex));

    // 입력을 복소수 배열에 제로 패딩하고 복사
    pad_real_to_complex<<<(padded+255)/256, 256>>>(d_X, d_x, N, padded);
    pad_real_to_complex<<<(padded+255)/256, 256>>>(d_H, d_h, K, padded);

    // 두 신호 모두 순방향 FFT
    cufftHandle plan;
    cufftPlan1d(&plan, padded, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_X, d_X, CUFFT_FORWARD);
    cufftExecC2C(plan, d_H, d_H, CUFFT_FORWARD);

    // 주파수 영역에서 요소별 곱셈
    complex_multiply<<<(padded+255)/256, 256>>>(d_X, d_H, padded);

    // 역 FFT
    cufftExecC2C(plan, d_X, d_X, CUFFT_INVERSE);

    // 실수 부분 추출 및 정규화
    extract_real_scaled<<<(M+255)/256, 256>>>(d_out, d_X, 1.0f/padded, M);

    cufftDestroy(plan);
    cudaFree(d_X); cudaFree(d_H);
}

// 주파수 영역 곱셈 kernel
__global__ void complex_multiply(cufftComplex *A, const cufftComplex *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float ar = A[i].x, ai = A[i].y;
    float br = B[i].x, bi = B[i].y;
    A[i].x = ar * br - ai * bi;   // A[i] * B[i]의 실수 부분
    A[i].y = ar * bi + ai * br;   // 허수 부분
}
```

**손익분기점**: N=10⁶일 때 K > log₂(N) ≈ 20이면 FFT 합성곱이 직접 합성곱보다 유리합니다.

---

## 7. 스펙트럼 필터링

고주파 계수를 0으로 만들어 저역 통과 필터를 적용합니다:

```c
// 저역 통과 필터: cutoff_frac * N/2 이하의 주파수만 유지
__global__ void lowpass_filter(cufftComplex *freq, int N, float cutoff_frac) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    // k를 부호 있는 주파수로 매핑 (-N/2 to N/2)
    int signed_k = (k <= N / 2) ? k : k - N;
    float normalized_freq = fabsf((float)signed_k) / (N / 2.0f);

    if (normalized_freq > cutoff_frac) {
        freq[k].x = 0.0f;
        freq[k].y = 0.0f;
    }
}

// 파워 스펙트럼: 각 주파수 bin에 대한 |X[k]|²
__global__ void power_spectrum(const cufftComplex *freq, float *power, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        power[k] = freq[k].x * freq[k].x + freq[k].y * freq[k].y;
    }
}
```

---

## 8. 파르스발 정리

에너지는 시간과 주파수 영역 사이에서 보존됩니다 (올바른 정규화로):

```
Σ |x[n]|² = (1/N) · Σ |X[k]|²

검증 코드 (정확성 확인용):
```

```c
// 순방향 FFT 후 파르스발 정리 검증
bool verify_parseval(const float *h_signal, const cufftComplex *h_freq, int N) {
    double energy_time = 0.0, energy_freq = 0.0;
    for (int n = 0; n < N; n++) energy_time += h_signal[n] * h_signal[n];
    for (int k = 0; k < N; k++) energy_freq +=
        h_freq[k].x * h_freq[k].x + h_freq[k].y * h_freq[k].y;
    energy_freq /= N;  // 정규화 팩터

    double ratio = energy_time / energy_freq;
    printf("Parseval 비율 (1.0이어야 함): %.6f\n", ratio);
    return fabs(ratio - 1.0) < 1e-4;
}
```

---

## 핵심 요약

- cuFFT는 비정규화 FFT를 계산합니다: 순방향 + 역방향 후 각 요소는 N이 곱해집니다 — `CUFFT_INVERSE` 후에는 항상 N으로 나누세요
- **R2C/C2R** 플랜은 에르미트 대칭을 활용하여 실수 값 신호에 대해 메모리와 런타임을 50% 절약합니다
- **배치 FFT** (batch > 1인 `cufftPlan1d` 또는 `cufftPlanMany`)는 많은 독립 변환에 올바른 API입니다
- **FFT 합성곱**은 O(N log N) vs 직접 O(N·K) — 손익분기점은 일반적으로 대형 N에서 K ≈ 20입니다
- 스펙트럼 필터링은 완전 병렬입니다: 순방향 FFT 후 원하지 않는 주파수 bin을 0으로 만들고 역변환 적용
- **파르스발 정리**는 저렴한 온전성 검사를 제공합니다: 시간 영역 에너지 = 주파수 영역 에너지/N

---

**다음**: [23. PDE Solvers Heat Equation](./23_PDE_Solvers_Heat_Equation.md) — 명시적 유한 차분으로 2D 열 방정식을 풀고, 안정 조건을 분석하며, GPU에서 다단계 시간 적분을 구현합니다.
