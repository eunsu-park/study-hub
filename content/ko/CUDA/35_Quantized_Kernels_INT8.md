# 35. Quantized Kernels — INT8

**이전**: [FlashAttention Kernel](./34_FlashAttention_Kernel.md) | **다음**: [Fused Kernel Patterns](./36_Fused_Kernel_Patterns.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. FP32 텐서를 INT8로 변환하는 absmax 양자화 구현하기
2. 텐서별 (활성화) 양자화와 채널별 (가중치) 양자화 구별하기
3. INT32 누적을 사용한 INT8 내적에 `dp4a` 명령 사용하기
4. 출력 에필로그에 역양자화가 융합된 INT8 GEMM kernel 설계하기
5. 메모리 대역폭 제한 추론을 위한 INT4 가중치 전용 역양자화 구현하기

---

## 1. 양자화가 필요한 이유

```
INT8 vs FP32 비교:
  저장:      4× 적은 메모리 (1 byte vs 4 bytes)
  대역폭:    메모리 트랜잭션당 4× 많은 값
  연산:      CUDA 코어에서 2× 처리량 (INT8 vs FP32)
             dp4a / Tensor Core 사용 시 4-8× 처리량 (INT8 TC)

A100 최대 처리량:
  FP32 CUDA 코어:   19.5 TFLOPS
  INT8 Tensor Core: 624 TOPS  (FP32 대비 32×!)

활용 사례:
  추론 양자화: 가중치 (INT4/INT8) + 활성화 (INT8/FP16)
  학습 양자화 (QAT): 학습 중 양자화 노이즈 시뮬레이션

정확도 손실:
  FP32 → INT8 가중치:      <0.5 perplexity 증가 (대형 모델)
  FP32 → INT4 가중치:      0.5-2 perplexity 증가
  FP32 → INT8 활성화:      신중한 보정 필요
```

---

## 2. Absmax 양자화

가장 단순한 균일 양자화: [-max_val, +max_val]을 [-127, 127]로 매핑:

```
scale       = max(|x|) / 127
x_quantized = round(x / scale)        [-127, 127]로 클램프
x_dequant   = x_quantized * scale

오류 분석:
  최대 양자화 오류 ≈ scale / 2 = max(|x|) / 254
  상대 오류 ≈ 1/254 ≈ 0.4%  (일반적인 가중치 분포에서)
```

```c
// float 배열을 INT8로 양자화 (absmax 텐서별)
__global__ void quantize_absmax(
    const float *x, int8_t *x_q, float *scale_out, int N)
{
    extern __shared__ float smem[];
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = (i < N) ? fabsf(x[i]) : 0.f;
    smem[tid] = val;
    __syncthreads();

    // block 내 최대값 reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    // thread 0이 atomicMax를 통해 block 최대값 게시 (int 표현 사용)
    if (tid == 0) {
        // 원자적 float 최대: IEEE 754 비트에서 정수 atomicMax 사용
        unsigned int old_bits = __float_as_uint(smem[0]);
        unsigned int *addr    = (unsigned int*)scale_out;
        atomicMax(addr, old_bits);
    }
    __syncthreads();

    // 전역 최대값이 사용 가능할 때까지 대기; 실제로는 2-패스 방식 사용
    // 최종 양자화 (scale = global_max / 127 사용)
    if (i < N) {
        float scale = (*scale_out) / 127.f;
        float q = __float2int_rn(x[i] / scale);   // 반올림
        x_q[i] = (int8_t)fminf(127.f, fmaxf(-127.f, q));
    }
}

// 역양자화: int8 → float
__global__ void dequantize(
    const int8_t *x_q, float *x_out, float scale, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x_out[i] = x_q[i] * scale;
}
```

---

## 3. 가중치의 채널별 양자화

채널별 양자화는 가중치 행렬의 각 출력 채널 (행)에 별도의 스케일을 할당합니다. 이는 채널별 값 분포 차이를 포착합니다:

```c
// 가중치 행렬 W [out_features × in_features] 양자화
// 각 행이 자체 스케일을 가짐 (출력 채널별)
__global__ void quantize_per_channel(
    const float *W,        // [OC × IC]
    int8_t *W_q,           // [OC × IC]
    float  *scales,        // [OC]
    int OC, int IC)
{
    int row = blockIdx.x;   // 출력 채널당 하나의 block
    int tid = threadIdx.x;

    // 이 행의 최대 절댓값 찾기
    float maxval = 0.f;
    for (int i = tid; i < IC; i += blockDim.x)
        maxval = fmaxf(maxval, fabsf(W[row * IC + i]));

    // block 수준 최대 reduction
    extern __shared__ float smax[];
    smax[tid] = maxval;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        __syncthreads();
    }

    float scale = smax[0] / 127.f;
    if (tid == 0) scales[row] = scale;
    __syncthreads();

    // 양자화
    for (int i = tid; i < IC; i += blockDim.x) {
        float q = W[row * IC + i] / scale;
        W_q[row * IC + i] = (int8_t)fminf(127.f, fmaxf(-127.f, __float2int_rn(q)));
    }
}
```

---

## 4. dp4a: 4회 연산으로 INT8 내적

`dp4a`는 단일 명령으로 4개 원소 INT8 내적을 INT32 누적으로 계산합니다:

```c
// dp4a: a = Σ_{k=0}^{3} a_k * b_k  (각 a_k, b_k는 int8; 결과는 int32)
// Pascal (sm_61) 이후 사용 가능

// 4개의 int8 값을 하나의 int32에 패킹
__device__ int pack_int8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((int)a & 0xFF) | (((int)b & 0xFF) << 8)
         | (((int)c & 0xFF) << 16) | (((int)d & 0xFF) << 24);
}

// 수동 dp4a (컴파일러는 보통 int8 로드로 자동 생성)
__device__ int dp4a(int a_packed, int b_packed, int c_acc) {
    return __dp4a(a_packed, b_packed, c_acc);
    // 내장 함수: int __dp4a(int a, int b, int c)
    //   반환: c + (a[7:0]*b[7:0]) + (a[15:8]*b[15:8])
    //              + (a[23:16]*b[23:16]) + (a[31:24]*b[31:24])
}

// K개 원소에 대한 INT8 내적 (K는 4의 배수여야 함)
__device__ int int8_dot(const int8_t *a, const int8_t *b, int K) {
    const int *a4 = (const int*)a;   // 패킹된 int8×4 배열로 처리
    const int *b4 = (const int*)b;
    int acc = 0;
    for (int k = 0; k < K/4; k++)
        acc = __dp4a(a4[k], b4[k], acc);
    return acc;  // INT32 누산기
}
```

---

## 5. 역양자화가 포함된 INT8 GEMM Kernel

```c
// INT8 GEMM: C_int32 = A_int8 · B_int8^T, 이후 역양자화
// A: [M × K] int8 (활성화, 텐서별 스케일 scale_a)
// B: [N × K] int8 (가중치, 채널별 스케일 scales_b[N])
// C: [M × N] float (역양자화 후 출력)

#define TILE 32

__global__ void int8_gemm(
    const int8_t *A, const int8_t *B,
    float *C,
    int M, int N, int K,
    float scale_a, const float *scales_b)
{
    __shared__ int8_t sA[TILE][TILE];
    __shared__ int8_t sB[TILE][TILE];  // B^T는 [N × K]로 저장 → tile [TILE × TILE]

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    int acc = 0;  // INT32 누산기

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // INT8 tile 로드 (dp4a를 위한 4-byte 정렬 로드)
        int a_col = t * TILE + tx;
        int b_col = t * TILE + ty;  // B는 [col][k]로 접근

        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
        sB[ty][tx] = (col < N && b_col < K) ? B[col * K + b_col] : 0;

        __syncthreads();

        // dp4a를 사용한 내적 (한 번에 4개 원소 패킹)
        // dp4a 정렬을 위해 TILE이 4의 배수인지 확인
        for (int k = 0; k < TILE; k += 4) {
            int a_packed = *reinterpret_cast<const int*>(&sA[ty][k]);
            int b_packed = *reinterpret_cast<const int*>(&sB[tx][k]);
            acc = __dp4a(a_packed, b_packed, acc);
        }

        __syncthreads();
    }

    // 역양자화 및 출력 쓰기
    if (row < M && col < N) {
        float scale = scale_a * scales_b[col];
        C[row * N + col] = (float)acc * scale;
    }
}
```

---

## 6. INT4 가중치 전용 역양자화

INT4 (4비트) 가중치는 바이트당 두 값을 패킹합니다. 메모리 대역폭을 절약하기 위해 GEMM 중에 즉석에서 역양자화합니다:

```c
// 하나의 uint8에 두 개의 INT4 값 패킹:
//   상위 니블 = 첫 번째 값 (bits 7:4)
//   하위 니블 = 두 번째 값 (bits 3:0)
// 범위: 부호 있는 INT4 → [-8, 7]

__device__ void unpack_int4x2(uint8_t packed, int8_t &hi, int8_t &lo) {
    // 4비트를 8비트로 부호 확장
    hi = (int8_t)((int8_t)(packed >> 4) << 4 >> 4);   // 산술적 우측 시프트
    lo = (int8_t)((int8_t)(packed << 4)       >> 4);
}

// INT4 가중치 역양자화 kernel
// W_int4: 패킹된 [OC × (IC/2)] uint8 (바이트당 2개 가중치)
// scales: [OC × (IC/group_size)] (그룹 양자화)
__global__ void dequant_int4_to_fp16(
    const uint8_t *W_int4,  // [OC × IC/2] 패킹됨
    __half *W_fp16,         // [OC × IC]   출력
    const __half *scales,   // [OC × ngroups] 그룹별 스케일
    int OC, int IC, int group_size)
{
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  // 반복당 두 개의 INT4
    if (oc >= OC || ic >= IC) return;

    // 패킹된 바이트 로드 (2개의 INT4 값)
    uint8_t packed = W_int4[oc * (IC/2) + ic/2];
    int8_t hi, lo;
    unpack_int4x2(packed, hi, lo);

    // 스케일: 그룹 양자화 (group_size 가중치당 하나의 스케일)
    int group = ic / group_size;
    float scale = __half2float(scales[oc * (IC / group_size) + group]);

    W_fp16[oc * IC + ic]   = __float2half(hi * scale);
    W_fp16[oc * IC + ic+1] = __float2half(lo * scale);
}
```

---

## 7. Perplexity 영향 및 보정

```
LLaMA-7B에 대한 양자화 정확도 (WikiText-2 perplexity):

정밀도          Perplexity  메모리 (7B 파라미터)
------------------------------------------------
FP32             5.68        28 GB
FP16             5.68        14 GB  (정확도 손실 없음)
INT8 (W8A8)      5.72        7 GB   (+0.04 ppl)
INT8 (W8A16)     5.70        7 GB   (+0.02 ppl)
INT4 (W4A16)     5.85        3.5 GB (+0.17 ppl)  ← 4× 메모리 감소!
INT4 NF4 (QLoRA) 5.80        3.5 GB (+0.12 ppl)  (NF4 = 정규화된 4비트)

보정 세트:
  약 512개의 대표적인 입력 샘플 수집
  활성화 범위 측정을 위한 순전파 실행
  실제 분포를 기반으로 채널별 스케일 계산
  나쁜 보정 → 좋은 보정 대비 정확도 10× 더 떨어짐
```

---

## 핵심 요약

- **Absmax 양자화**: `scale = max(|x|) / 127`, `x_q = round(x / scale)`을 [-127, 127]로 클램프; `x = x_q * scale`로 역양자화
- **텐서별 vs 채널별**: 활성화는 텐서별 스케일 사용 (컴파일 시 알 수 없음); 가중치는 더 나은 정확도를 위해 채널별 스케일 사용 (출력 행당 하나의 스케일)
- **dp4a** (`__dp4a`): 단일 명령으로 INT32 누적을 사용한 4개 원소 INT8 내적 계산 — GPU에서 INT8 GEMM의 기본 구성 요소
- **INT8 GEMM 에필로그**: INT32 누산기는 쓰기 전에 역양자화 필요: `C_float[i] = acc_int32 * scale_A * scale_B[col]`
- **INT4 가중치 전용**: 바이트당 2개의 가중치 패킹; 행렬 곱 전에 즉석에서 FP16으로 역양자화; FP16 가중치 대비 4× 메모리 대역폭 절약
- **보정**이 중요: 모델을 통해 대표적인 데이터를 실행하여 실제 활성화 범위 측정; 불량한 보정은 이론적 최솟값보다 5-10× 더 큰 정확도 손실 유발

---

**다음**: [36. Fused Kernel Patterns](./36_Fused_Kernel_Patterns.md) — kernel 융합이 왜 메모리 왕복 횟수를 줄이는지 알아보고, 융합 bias+ReLU, 융합 residual+LayerNorm 및 기타 일반적인 딥러닝 kernel 융합 패턴을 구현합니다.
