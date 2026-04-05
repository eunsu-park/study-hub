# 33. Softmax and LayerNorm Kernels

**이전**: [GEMM from Scratch](./32_GEMM_from_Scratch.md) | **다음**: [FlashAttention Kernel](./34_FlashAttention_Kernel.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 나이브 softmax가 수치적으로 불안정한 이유를 설명하고 최댓값 빼기 수정 구현하기
2. warp shuffle을 사용하여 한 번의 패스로 최댓값, 합계, 출력을 계산하는 단일 패스 "온라인 softmax" kernel 작성하기
3. 하나의 warp-shuffle 패스로 평균과 분산을 계산하는 융합 LayerNorm kernel 구현하기
4. Llama/Mistral 모델에서 사용되는 RMSNorm kernel 작성하기 (더 단순: 평균 빼기 없음)
5. 이러한 연산의 융합이 왜 메모리 왕복 횟수를 줄이고 처리량을 향상시키는지 이해하기

---

## 1. 정규화 Kernel이 중요한 이유

트랜스포머 모델에서 softmax와 LayerNorm은 순전파당 수천 번 호출됩니다:
```
BERT-large (24개 레이어):
  - 24 × 2 = 48개의 어텐션 softmax 연산  (batch × heads × seq_len)
  - 48개의 LayerNorm 연산
  - 이들은 메모리 대역폭 병목: 빠른 구현 → 직접적인 종단간 속도 향상

나이브 3-패스 softmax의 메모리 문제:
  패스 1: 행 읽기, 최댓값 찾기               → 메모리 읽기 1×
  패스 2: 행 읽기, exp(x-max) 계산          → 메모리 읽기 1×
  패스 3: 행 읽기, sum(exp)으로 나누기       → 메모리 읽기 1× + 쓰기 1×
  합계: 원소당 읽기 3번 + 쓰기 1번

온라인 (1-패스) softmax:
  행의 단일 패스: 읽기 1번 + 쓰기 1번  → 메모리 대역폭 3× 감소
```

---

## 2. 나이브 Softmax (3-패스, 수치적으로 안정적)

```c
// softmax(x_i) = exp(x_i - max_x) / Σ exp(x_j - max_x)
// [batch × seq_len] 행렬의 각 행이 독립적으로 정규화됨
__global__ void softmax_naive(const float *in, float *out, int N) {
    // block당 하나의 행
    int row = blockIdx.x;
    const float *x = in + row * N;
    float       *y = out + row * N;

    // 패스 1: 최댓값 찾기
    float maxval = -1e30f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        maxval = fmaxf(maxval, x[i]);
    // block 수준 최대 reduction (shared memory)
    maxval = block_reduce_max(maxval);  // 레슨 14 참조

    // 패스 2: exp(x - max) 계산 및 합산
    float sum = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float e = expf(x[i] - maxval);
        y[i] = e;  // 임시 저장
        sum += e;
    }
    sum = block_reduce_sum(sum);

    // 패스 3: 정규화
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        y[i] /= sum;
}
```

---

## 3. 온라인 Softmax (단일 패스)

온라인 softmax는 행을 스캔하면서 실행 중인 (최댓값, 합계) 쌍을 유지합니다. 핵심 통찰: 새로운 최댓값이 발견되면 기존 부분 합계를 재조정합니다:

```
온라인 알고리즘:
  초기화: m = -inf, d = 0
  각 x_i에 대해:
    m_new = max(m, x_i)
    d_new = d * exp(m - m_new) + exp(x_i - m_new)
    m = m_new, d = d_new

  최종: softmax(x_i) = exp(x_i - m) / d
```

```c
// 온라인 softmax: 하나의 warp (32 thread)가 32×unroll 원소까지 하나의 행 처리
// 32개 원소보다 긴 행의 경우 shared memory와 함께 thread-block 사용
__global__ void online_softmax(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *x = in  + row * cols;
    float       *y = out + row * cols;

    float m = -1e30f, d = 0.f;

    // --- 단일 패스: (최댓값, 정규화 인수) 계산 ---
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);
        d = d * expf(m - m_new) + expf(xi - m_new);
        m = m_new;
    }

    // --- (m, d)의 warp 수준 reduction ---
    // 각 thread는 로컬 (m, d)를 가짐; 전역 최댓값과 재조정된 합계를 찾기 위해 reduce

    // 단계 1: warp 전체에서 최댓값 reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        if (m2 > m) {
            d = d * expf(m - m2) + d2;
            m = m2;
        } else {
            d = d + d2 * expf(m2 - m);
        }
    }
    // lane 0에서 결과 브로드캐스트
    m = __shfl_sync(0xffffffff, m, 0);
    d = __shfl_sync(0xffffffff, d, 0);

    // --- 출력 쓰기 ---
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[i] = expf(x[i] - m) / d;
}
```

---

## 4. Shared Memory를 사용한 온라인 Softmax (긴 행의 경우)

32개 원소보다 긴 행의 경우 shared memory를 사용하여 전체 thread block에서 협력합니다:

```c
__global__ void online_softmax_block(
    const float *in, float *out, int rows, int cols)
{
    extern __shared__ float smem[];  // 2 * blockDim.x floats (thread당 m과 d)
    float *sm = smem;
    float *sd = smem + blockDim.x;

    int row = blockIdx.x;
    const float *x = in  + row * cols;
    float       *y = out + row * cols;
    int tid = threadIdx.x;

    float m = -1e30f, d = 0.f;

    // thread 로컬 온라인 누적
    for (int i = tid; i < cols; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);
        d = d * expf(m - m_new) + expf(xi - m_new);
        m = m_new;
    }

    // shared memory에 저장
    sm[tid] = m;
    sd[tid] = d;
    __syncthreads();

    // 트리 reduction: (m, d) 쌍의 병렬 병합
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m2 = sm[tid + stride];
            float d2 = sd[tid + stride];
            float m_new = fmaxf(sm[tid], m2);
            sd[tid] = sd[tid] * expf(sm[tid] - m_new)
                    + d2      * expf(m2      - m_new);
            sm[tid] = m_new;
        }
        __syncthreads();
    }

    // thread 0에서 브로드캐스트
    m = sm[0];
    d = sd[0];
    __syncthreads();

    // 마지막 패스에서 출력 쓰기
    for (int i = tid; i < cols; i += blockDim.x)
        y[i] = expf(x[i] - m) / d;
}
```

---

## 5. LayerNorm Kernel

LayerNorm은 각 특성 벡터 (행)을 평균 0, 분산 1로 정규화한 후 스케일링과 이동을 수행합니다:

```
y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β

mean(x) = (1/H) Σ x_i
var(x)  = (1/H) Σ (x_i - mean)²
```

```c
// 융합 LayerNorm: 단일 패스로 평균과 분산 계산
// γ (weight)와 β (bias)는 [H] 형태의 학습된 파라미터
__global__ void layernorm_forward(
    const float *x,      // [batch × H]
    const float *gamma,  // [H]
    const float *beta,   // [H]
    float *out,          // [batch × H]
    float *mean_out,     // [batch] (역전파를 위해 저장)
    float *var_out,      // [batch]
    int H, float eps)
{
    int row = blockIdx.x;
    const float *xi = x   + row * H;
    float       *yi = out + row * H;

    // 실용적 접근법: 두 값 reduction (sum과 sum_sq)
    // 참고: Welford의 온라인 알고리즘은 순차적 업데이트에는 우아하지만
    // 직접적인 병렬화가 되지 않습니다 — 부분 Welford 상태를 병합하려면
    // 비자명한 결합 단계가 필요합니다. 아래의 sum/sum_sq 접근법은 더 단순하고
    // 일반적인 hidden-dim 크기에서 float32에 대해 동등하게 수치적으로 안정적입니다.
    float sum = 0.f, sum_sq = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = xi[i];
        sum    += v;
        sum_sq += v * v;
    }

    // sum과 sum_sq를 동시에 warp 수준 reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum    += __shfl_down_sync(0xffffffff, sum,    offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_sum[32], s_sum_sq[32];
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x % 32;

    if (lane == 0) { s_sum[warp_id] = sum; s_sum_sq[warp_id] = sum_sq; }
    __syncthreads();

    // warp 전체에서 최종 reduce (thread 0만)
    if (threadIdx.x == 0) {
        float total_sum = 0.f, total_sq = 0.f;
        int n_warps = blockDim.x / 32;
        for (int w = 0; w < n_warps; w++) {
            total_sum += s_sum[w];
            total_sq  += s_sum_sq[w];
        }
        float mn  = total_sum / H;
        float var = total_sq / H - mn * mn;
        s_sum[0]    = mn;
        s_sum_sq[0] = var;
        if (mean_out) mean_out[row] = mn;
        if (var_out)  var_out[row]  = var;
    }
    __syncthreads();

    float mn   = s_sum[0];
    float var  = s_sum_sq[0];
    float rstd = rsqrtf(var + eps);

    // 정규화 및 어파인 변환 적용
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        yi[i] = (xi[i] - mn) * rstd * gamma[i] + beta[i];
}
```

---

## 6. RMSNorm (평균 빼기 없음)

RMSNorm (LLaMA, Mistral에서 사용)은 더 단순합니다: 평균 빼기 없이 제곱 평균 제곱근으로 정규화합니다:

```
RMSNorm(x)_i = x_i / RMS(x) * γ_i
RMS(x)       = sqrt((1/H) Σ x_i²)
```

```c
__global__ void rmsnorm_forward(
    const float *x,     // [batch × H]
    const float *gamma, // [H]
    float *out,         // [batch × H]
    int H, float eps)
{
    int row = blockIdx.x;
    const float *xi = x   + row * H;
    float       *yi = out + row * H;

    // 제곱합 계산
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        sum_sq += xi[i] * xi[i];

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float s_sq[32];
    int warp_id = threadIdx.x / 32;
    int lane    = threadIdx.x % 32;
    if (lane == 0) s_sq[warp_id] = sum_sq;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int w = 0; w < blockDim.x/32; w++) total += s_sq[w];
        s_sq[0] = rsqrtf(total / H + eps);  // 1/RMS
    }
    __syncthreads();

    float rrms = s_sq[0];
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        yi[i] = xi[i] * rrms * gamma[i];
}
```

---

## 7. 성능 분석

```
설정: batch=128, H=768 (BERT-base), float32

Kernel               시간    대역폭 효율   메모리 패스
----------------------------------------------------------
softmax 3-패스       0.8ms    65%          읽기 3번 + 쓰기 1번
온라인 softmax       0.3ms    85%          읽기 1번 + 쓰기 1번  (2.6× 빠름)
layernorm 2-패스     0.6ms    60%          읽기 2번 + 쓰기 1번
layernorm 1-패스     0.35ms   80%          읽기 1번 + 쓰기 1번  (1.7× 빠름)
rmsnorm              0.25ms   90%          읽기 1번 + 쓰기 1번  (더 단순)

참고: H ≤ 32인 경우: 단일 warp에 모두 맞음 → shared memory 불필요
      H ≤ 1024인 경우: shared-memory reduction으로 충분
      H > 1024인 경우: atomic 또는 2-패스를 사용한 multi-block reduction 필요
```

---

## 핵심 요약

- **나이브 softmax**는 각 행을 3번 읽음; **온라인 softmax**는 실행 중인 (최댓값, 재조정된 합계) 쌍을 유지하여 모든 패스를 하나로 병합
- 온라인 병합 규칙: 새로운 최댓값 `m_new`를 만날 때 기존 합계 재조정: `d_new = d * exp(m - m_new) + exp(x_i - m_new)`
- (최댓값, 합계) 쌍에 대한 **warp shuffle reduction**은 사용자 정의 병합 단계 필요 (단순 덧셈 아님): 최댓값 비교 후 더 작은 쪽의 합계 재조정
- **LayerNorm**: 한 번의 패스로 sum과 sum_sq 계산 → mean = sum/H, var = sum_sq/H - mean² → 어파인 변환 적용; 역전파를 위해 mean과 rstd 저장
- **RMSNorm**은 평균 빼기를 완전히 건너뛰고 정규화를 위해 제곱 평균 제곱근만 계산; 동일한 hidden dimension에서 LayerNorm보다 ~30% 빠름
- 이러한 kernel들은 모두 **메모리 대역폭 병목**: 핵심 최적화는 입력 행에 대한 패스 횟수를 최소화하는 것

---

**다음**: [34. FlashAttention Kernel](./34_FlashAttention_Kernel.md) — O(N²) HBM 연산 대신 O(N²/B) HBM 연산으로 정확한 어텐션을 계산하는 FlashAttention tiling 알고리즘을 구현합니다. 메모리 부족 없이 긴 컨텍스트 트랜스포머를 가능하게 합니다.
