# 36. Fused Kernel Patterns

**이전**: [Quantized Kernels INT8](./35_Quantized_Kernels_INT8.md) | **다음**: [Multi-GPU and NCCL](./37_Multi_GPU_and_NCCL.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 메모리 병목 kernel이 GPU 대역폭을 낭비하는 방법과 융합이 왜 도움이 되는지 설명하기
2. 별도의 메모리 패스를 피하는 융합 bias+ReLU kernel 구현하기
3. 단일 패스로 융합 residual+LayerNorm kernel 구현하기
4. tanh 근사를 사용한 융합 bias+GELU kernel 작성하기
5. 루프라인 모델을 사용하여 융합 vs 비융합 kernel의 대역폭 절약 정량화하기

---

## 1. Kernel 융합이 필요한 이유

모든 별도 kernel 실행은 HBM (GPU 메인 메모리)에서 입력을 읽고 출력을 써야 합니다. 짧은 원소별 연산의 경우 산술은 간단하지만 병목은 메모리입니다:

```
예시: GEMM 출력 → bias 추가 → ReLU (비융합)

융합 없이:                               원소당 HBM 트래픽
  1. GEMM이 C[M×N]을 HBM에 씀             → 쓰기 1×
  2. bias_add가 C를 읽고 C+b를 씀        → 읽기 1× + 쓰기 1×
  3. relu가 C+b를 읽고 max(0,C+b)를 씀  → 읽기 1× + 쓰기 1×
  합계: 읽기 3번 + 쓰기 3번 = 6 × M × N × 4 bytes

융합 사용 (하나의 에필로그 kernel):
  1. bias+ReLU가 C를 한 번 읽고 한 번 씀  → 읽기 1× + 쓰기 1×
  합계: 읽기 1번 + 쓰기 1번 = 2 × M × N × 4 bytes  (3× 적은 트래픽)

M=N=4096, FP32: 3× = GEMM block당 192 MB 절약
900 GB/s HBM 대역폭에서: GEMM당 0.21 ms 절약
```

---

## 2. 융합 Bias + ReLU

```c
// GEMM 출력이 C[M×N]에 저장됨; bias b[N] (출력 열당 하나)
// 인플레이스 적용: C[row][col] = max(0, C[row][col] + b[col])

__global__ void bias_relu_inplace(float *C, const float *bias, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float val = C[row * N + col] + bias[col];
    C[row * N + col] = fmaxf(0.f, val);   // ReLU
}

// 벡터화 버전: 한 번에 4개 열 처리
__global__ void bias_relu_vec4(float *C, const float *bias, int M, int N) {
    int col4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int row  =  blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col4 + 3 >= N) return;

    // 단일 128비트 로드로 4개의 bias 값과 4개의 C 값 로드
    float4 b4 = *reinterpret_cast<const float4*>(bias + col4);
    float4 c4 = *reinterpret_cast<const float4*>(C + row * N + col4);

    c4.x = fmaxf(0.f, c4.x + b4.x);
    c4.y = fmaxf(0.f, c4.y + b4.y);
    c4.z = fmaxf(0.f, c4.z + b4.z);
    c4.w = fmaxf(0.f, c4.w + b4.w);

    *reinterpret_cast<float4*>(C + row * N + col4) = c4;
}
```

---

## 3. 융합 Bias + GELU

GELU (Gaussian Error Linear Unit)는 BERT, GPT-2 및 많은 현대 트랜스포머에서 사용됩니다. tanh 근사는 비용이 높은 `erff()`를 피합니다:

```
정확한 GELU:        x * 0.5 * (1 + erf(x / sqrt(2)))
근사 GELU:          x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
                    = x * 0.5 * (1 + tanh(0.7978845 * (x + 0.044715 * x³)))

오류: 모든 x에서 < 0.001; PyTorch의 F.gelu(approximate='tanh')에서 사용
```

```c
__device__ __forceinline__ float gelu_approx(float x) {
    const float k0 = 0.7978845608f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.f + tanhf(inner));
}

__global__ void bias_gelu_inplace(float *C, const float *bias, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float val = C[row * N + col] + bias[col];
    C[row * N + col] = gelu_approx(val);
}

// GELU 역전파 (학습용):
// d_GELU/dx = 0.5*(1+tanh(inner)) + 0.5*x*(1-tanh²(inner))*d(inner)/dx
__device__ __forceinline__ float gelu_approx_grad(float x) {
    const float k0 = 0.7978845608f, k1 = 0.044715f;
    float inner = k0 * (x + k1 * x * x * x);
    float tanh_v = tanhf(inner);
    float dtanh  = 1.f - tanh_v * tanh_v;
    float dinner = k0 * (1.f + 3.f * k1 * x * x);
    return 0.5f * (1.f + tanh_v) + 0.5f * x * dtanh * dinner;
}
```

---

## 4. 융합 Residual + LayerNorm

트랜스포머 block에서 LayerNorm은 일반적으로 잔차 연결 후에 적용됩니다:
`y = LayerNorm(x + residual)`

융합을 통해 중간 `x + residual`을 HBM에 쓰는 것을 피합니다:

```c
__global__ void fused_residual_layernorm(
    const float *x,        // [batch × H] — 메인 스트림
    const float *res,      // [batch × H] — 추가할 residual
    const float *gamma,    // [H] weight
    const float *beta,     // [H] bias
    float *out,            // [batch × H] 출력
    float *mean_out,       // [batch] 저장된 평균 (역전파용)
    float *rstd_out,       // [batch] 저장된 1/std
    int H, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float *xi  = x   + row * H;
    const float *ri  = res + row * H;
    float       *yi  = out + row * H;

    // 패스 1: (x + residual)의 sum과 sum_sq 계산
    float sum = 0.f, sum_sq = 0.f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = xi[i] + ri[i];   // residual 추가 (중간 쓰기 없음)
        sum    += v;
        sum_sq += v * v;
    }

    // Warp 수준 reduction
    for (int off = 16; off > 0; off >>= 1) {
        sum    += __shfl_down_sync(0xffffffff, sum,    off);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
    }

    __shared__ float s_sum[32], s_sq[32];
    int warp = tid / 32, lane = tid % 32;
    if (lane == 0) { s_sum[warp] = sum; s_sq[warp] = sum_sq; }
    __syncthreads();

    if (tid == 0) {
        float ts = 0.f, tsq = 0.f;
        int nw = blockDim.x / 32;
        for (int w = 0; w < nw; w++) { ts += s_sum[w]; tsq += s_sq[w]; }
        float mn  = ts / H;
        float var = tsq / H - mn * mn;
        float rs  = rsqrtf(var + eps);
        s_sum[0] = mn;
        s_sq[0]  = rs;
        if (mean_out) mean_out[row] = mn;
        if (rstd_out) rstd_out[row] = rs;
    }
    __syncthreads();

    float mn = s_sum[0], rstd = s_sq[0];

    // 패스 2: LayerNorm 변환 적용
    for (int i = tid; i < H; i += blockDim.x) {
        float v = xi[i] + ri[i];   // 재계산 (shared에 저장하는 것보다 저렴)
        yi[i] = (v - mn) * rstd * gamma[i] + beta[i];
    }
}
```

---

## 5. 융합 Dropout + Add

Dropout은 랜덤 활성화를 0으로 만들고 나머지를 1/(1-p)로 스케일합니다. residual 추가와 융합하면 두 번의 메모리 패스가 절약됩니다:

```c
// 융합 dropout + residual add
// out = dropout(x, p) + residual
__global__ void fused_dropout_add(
    const float *x, const float *residual,
    float *out, uint8_t *mask_out,   // 역전파를 위해 마스크 저장
    int N, float p, curandState *states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState local = states[i];
    float u    = curand_uniform(&local);
    states[i]  = local;

    float keep  = (u >= p) ? 1.f : 0.f;   // 1 = 유지, 0 = 드롭
    float scale = 1.f / (1.f - p);        // 역 dropout 스케일링

    float val = x[i] * keep * scale + residual[i];
    out[i]     = val;
    mask_out[i] = (uint8_t)keep;
}
```

---

## 6. CUTLASS 에필로그 융합 (개념)

CUTLASS (선형 대수 서브루틴을 위한 CUDA 템플릿)는 사용자 정의 가능한 에필로그로 GEMM을 구현합니다:

```cpp
// CUTLASS 에필로그 개념: GEMM 출력에 원소별 변환 적용
// GEMM과 에필로그 사이에 전체 행렬을 HBM에 쓰지 않음

// 에필로그 연산 정의 (bias + ReLU)
using BiasReluEpilogue = cutlass::epilogue::threadblock::LinearCombinationRelu<
    cutlass::half_t,        // 원소 타입
    4,                      // 벡터 저장당 원소 수 (float4 동등)
    float,                  // 누산기 타입
    float,                  // 스케일 인수 타입
    cutlass::epilogue::threadblock::ScaleType::NoBetaScaling
>;

// 에필로그는 GEMM과 동일한 kernel에서 실행되며, register에서 C를 읽음
// (GEMM 출력과 bias+ReLU 사이에 HBM에 절대 접근하지 않음)
// 이것이 cuBLAS가 융합 연산에 내부적으로 사용하는 기법

// CUTLASS 에필로그 방문자 패턴 (CUTLASS 3.x):
// GEMM 에필로그에 융합되는 원소별 연산의 계산 그래프 정의
//   계산: (alpha * A*B + beta * C) → bias_add → gelu → 출력
```

---

## 7. 성능 비교

```
설정: M=4096, N=4096, A100에서 FP32

연산 시퀀스: GEMM → bias_add → ReLU
접근법              시간    HBM 읽기   HBM 쓰기
------------------------------------------------------
별도 kernel         3.1ms   192 MB     192 MB     (MN에 대한 추가 패스 3회)
융합 에필로그       2.2ms    64 MB      64 MB     (패스 1회, 레지스터 내)
절약:               1.9ms   132 MB     132 MB     (메모리 트래픽 2.8× 감소)

참고: GEMM 자체는 1.8ms; 별도 원소별 연산이 72% 오버헤드 추가
      융합 사용 시: 22% 오버헤드만 → 총 2.2ms vs 3.1ms

대역폭 예산:
  비융합: 3 × M × N × 4 bytes = 192 MB, 900 GB/s에서 = 낭비 0.21 ms
  융합:   1 × M × N × 4 bytes = 64 MB                → 0.14 ms 절약

24-레이어 BERT의 경우:
  24 × 4 GEMM block × 0.9ms 절약 = 순전파당 총 86ms 속도 향상
```

---

## 8. 일반적인 융합 지침

```
융합에 적합한 후보:
  - GEMM/Conv 후 원소별 연산 (bias, 활성화, dropout, LayerNorm)
  - 브로드캐스트가 뒤따르는 reduction (평균 빼기, softmax)
  - 중간 결과가 register에 맞는 읽기-수정-쓰기 체인

융합에 적합하지 않은 후보:
  - 두 개의 GEMM (연산 병목; 융합이 대역폭 도움 안 됨)
  - 매우 다른 병렬성을 가진 연산 (예: 1D + 2D kernel)
  - 그 사이에 전역 동기화가 필요한 연산

결정 규칙 (루프라인):
  대역폭 제한인 경우: 융합 (메모리 패스 제거)
  연산 제한인 경우:   신경 쓰지 않음 (메모리 절약이 도움 안 됨)

구현 순서:
  1. ncu로 프로파일 → 대역폭 병목 연산 식별
  2. 해당 연산 융합 → 재프로파일
  3. 정확성 검증 (특히 역전파의 경우)
```

---

## 핵심 요약

- 모든 별도 kernel 패스는 HBM에서 전체 텐서를 읽고 씀; 두 연산을 하나의 kernel에 융합하면 대역폭 병목 워크로드의 HBM 트래픽이 절반으로 감소
- **융합 bias+ReLU**: 단일 인플레이스 kernel; 4× 메모리 처리량을 위해 float4 로드 사용
- **근사 GELU**: `0.5 * x * (1 + tanh(0.7978 * (x + 0.044715 * x³)))`는 널리 사용되며 정확한 GELU 대비 <0.1% 오류
- **융합 residual+LayerNorm**: HBM에 쓰는 대신 `x + res`를 두 번 계산 (통계 한 번, 출력 한 번) — register 재계산이 HBM 왕복보다 빠름
- **CUTLASS 에필로그**: GEMM 출력이 에필로그 전체에서 register/shared memory에 남아 있어, HBM 비용 없는 bias/활성화 융합 가능
- 융합은 **대역폭 병목** 연산에만 적용; 두 연산 병목 연산의 융합은 이점이 없고 코드 복잡성만 증가

---

**다음**: [37. Multi-GPU and NCCL](./37_Multi_GPU_and_NCCL.md) — CUDA peer-to-peer 전송과 NCCL collective 연산을 사용하여 단일 GPU를 넘어 데이터, 텐서, 파이프라인 병렬성을 구현합니다.
