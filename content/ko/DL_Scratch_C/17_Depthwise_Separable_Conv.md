# 17. 깊이별 분리 합성곱

**이전**: [ResNet과 스킵 연결](./16_ResNet_and_Skip_Connections.md) | **다음**: [스퀴즈-익사이테이션과 어텐션](./18_Squeeze_Excitation_and_Attention.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 깊이별 분리 합성곱(depthwise separable convolution)이 표준 합성곱을 어떻게 인수분해하는지 설명하기
2. 표준 합성곱과 비교한 정확한 FLOP 감소량 계산하기
3. C 언어로 깊이별 합성곱의 순전파와 역전파 구현하기
4. 점별(pointwise) 1×1 합성곱을 행렬 곱으로 구현하기
5. MobileNet 스타일의 역전 잔차 블록(inverted residual block) 구축하기

---

## 1. 표준 합성곱 vs 깊이별 분리 합성곱

### 표준 합성곱

```
입력:  [N, C_in, H, W]
필터: [C_out, C_in, K, K]
출력:  [N, C_out, OH, OW]

FLOPs = N × C_out × OH × OW × C_in × K × K × 2
      = 2 × N × OH × OW × C_out × C_in × K²
```

### 깊이별 분리 합성곱

두 단계로 분리:

```
1단계 — 깊이별 합성곱(depthwise conv): 각 채널을 독립적으로 필터링
  입력:  [N, C_in, H, W]
  필터: [C_in, 1, K, K]  (입력 채널당 하나의 K×K 필터, 채널 간 혼합 없음)
  출력:  [N, C_in, OH, OW]
  FLOPs = 2 × N × C_in × OH × OW × K²

2단계 — 점별 합성곱(pointwise conv, 1×1 conv): 채널 혼합
  입력:  [N, C_in, OH, OW]
  필터: [C_out, C_in, 1, 1]
  출력:  [N, C_out, OH, OW]
  FLOPs = 2 × N × C_out × OH × OW × C_in
```

### FLOP 비율

```
깊이별 분리 합성곱(DWS) 총 FLOPs = 2 × N × OH × OW × C_in × (K² + C_out)

표준 합성곱 대비 비율:
  DWS / 표준 = (K² + C_out) / (C_in × K² / C_in × C_out... 다시 계산하면)

표준: 2 × N × OH × OW × C_out × C_in × K²
DWS: 2 × N × OH × OW × C_in × K² + 2 × N × OH × OW × C_out × C_in
   = 2 × N × OH × OW × C_in × (K² + C_out)

비율 = (K² + C_out) / (C_out × K²)
     = 1/C_out + 1/K²

K=3, C_out=256:  1/256 + 1/9 ≈ 0.115  →  8.7배 FLOP 감소
K=3, C_out=128:  1/128 + 1/9 ≈ 0.119  →  8.4배 FLOP 감소
K=5, C_out=256:  1/256 + 1/25 ≈ 0.044 → 22.8배 FLOP 감소
```

---

## 2. 깊이별 합성곱

### 순전파

```c
// depthwise_conv2d_forward: 각 채널이 자체 [KH, KW] 필터를 가짐
// 입력:   [N, C, H, W]
// 가중치:  [C, 1, KH, KW]  [C, KH*KW]로 저장
// 출력:  [N, C, OH, OW]
void depthwise_conv2d_forward(
    const float *input,   // [N, C, H, W]
    const float *weight,  // [C, KH, KW]  (채널당 하나의 필터)
    const float *bias,    // [C]  (NULL일 수 있음)
    float       *output,  // [N, C, OH, OW]
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float sum = bias ? bias[c] : 0.0f;
        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                sum += NCHW(input, N, C, H, W, n, c, ih, iw)
                     * weight[c * KH * KW + kh * KW + kw];
        }
        NCHW(output, N, C, OH, OW, n, c, oh, ow) = sum;
    }
}
```

### 역전파

```c
// depthwise_conv2d_backward: dX, dW, db 계산
void depthwise_conv2d_backward(
    const float *input,   // [N, C, H, W]
    const float *weight,  // [C, KH, KW]
    const float *dY,      // [N, C, OH, OW]
    float       *dX,      // [N, C, H, W]   — 0으로 초기화
    float       *dW,      // [C, KH, KW]    — 0으로 초기화
    float       *db,      // [C]             — 0으로 초기화 (NULL일 수 있음)
    int N, int C, int H, int W,
    int KH, int KW, int OH, int OW,
    int stride, int pad) {

    for (int n  = 0; n  < N;  n++)
    for (int c  = 0; c  < C;  c++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float grad = NCHW(dY, N, C, OH, OW, n, c, oh, ow);
        if (db) db[c] += grad;

        for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float x_val = NCHW(input, N, C, H, W, n, c, ih, iw);
                // dW
                dW[c * KH * KW + kh * KW + kw] += grad * x_val;
                // dX
                NCHW(dX, N, C, H, W, n, c, ih, iw)
                    += grad * weight[c * KH * KW + kh * KW + kw];
            }
        }
    }
}
```

---

## 3. 점별 합성곱 (1×1 conv)

1×1 합성곱은 채널 차원에 걸친 행렬 곱과 동일합니다:

```c
// pointwise_conv2d: 1×1 conv = 각 공간 위치에서 채널에 걸친 행렬 곱
// 내부적으로: [N, C_in, H, W] → [N*H*W, C_in]으로 재구성
//            가중치 [C_out, C_in]^T와 행렬 곱
void pointwise_conv2d(
    const float *input,   // [N, C_in, H, W]
    const float *weight,  // [C_out, C_in]
    const float *bias,    // [C_out]
    float       *output,  // [N, C_out, H, W]
    int N, int C_in, int C_out, int H, int W) {

    int M = N * H * W;  // 공간 위치 수

    // 입력을 [N*H*W, C_in]으로 재해석 — 행렬 곱을 위해 NHWC 형태로 전환
    // 전략: NHWC로 전치 후 행렬 곱, 다시 NCHW로 전치
    float *X_nhwc = malloc(M * C_in * sizeof(float));
    float *Y_nhwc = malloc(M * C_out * sizeof(float));

    // NCHW → NHWC
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C_in; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        X_nhwc[n * H * W * C_in + h * W * C_in + w * C_in + c]
            = NCHW(input, N, C_in, H, W, n, c, h, w);

    // 행렬 곱: [M, C_in] × [C_in, C_out] → [M, C_out]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, C_out, C_in,
                1.0f, X_nhwc, C_in,
                weight, C_in,
                0.0f, Y_nhwc, C_out);

    // 편향 추가
    if (bias) {
        for (int i = 0; i < M; i++)
        for (int c = 0; c < C_out; c++)
            Y_nhwc[i * C_out + c] += bias[c];
    }

    // NHWC → NCHW
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
    for (int c = 0; c < C_out; c++)
        NCHW(output, N, C_out, H, W, n, c, h, w)
            = Y_nhwc[n * H * W * C_out + h * W * C_out + w * C_out + c];

    free(X_nhwc); free(Y_nhwc);
}
```

---

## 4. MobileNet 스타일의 역전 잔차 블록

MobileNetV2는 "역전 잔차(inverted residuals)"를 사용합니다: 채널 확장 → 깊이별 합성곱 → 압축:

```
표준 잔차 (ResNet):
  넓음 → 좁음 → 넓음  (병목이 압축)

역전 잔차 (MobileNetV2):
  좁음 → 넓음 → 좁음  (확장 후 압축)

구조:
  x → PW(t×C_in) → BN → ReLU6 → DW(stride) → BN → ReLU6 → PW(C_out) → BN
  ↓_____________________________________________________↑ (stride=1이고 C_in==C_out인 경우)

여기서 t = 확장 계수 (일반적으로 6)
```

```c
// ReLU6: 활성화를 [0, 6]으로 클램프 — 낮은 정밀도에서 큰 활성화 방지
void relu6_forward(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = fmaxf(0.0f, fminf(6.0f, X[i]));
}

typedef struct {
    int C_in, C_mid, C_out, stride;
    float *pw1_w;   // [C_mid, C_in]       확장
    float *dw_w;    // [C_mid, KH, KW]     깊이별
    float *pw2_w;   // [C_out, C_mid]      투영
    // 각 서브 레이어에 대한 BN...
    int use_residual;  // stride=1이고 C_in==C_out이면 1
} InvertedResidual;

// 역전 잔차 블록의 FLOP 수 계산
long inverted_residual_flops(int C_in, int C_out, int H, int W, int t, int K, int stride) {
    int C_mid = C_in * t;
    int OH = (H + stride - 1) / stride;
    int OW = (W + stride - 1) / stride;
    long pw1_flops = 2L * H * W * C_mid * C_in;
    long dw_flops  = 2L * OH * OW * C_mid * K * K;
    long pw2_flops = 2L * OH * OW * C_out * C_mid;
    return pw1_flops + dw_flops + pw2_flops;
}
```

---

## 5. MobileNet vs ResNet: FLOP 비교

```
작업: [1, 128, 56, 56] → [1, 256, 56, 56]의 3×3 합성곱

표준 합성곱:  2 × 256 × 56 × 56 × 128 × 9 = 924M FLOPs
DWS 합성곱:   2 × 128 × 56 × 56 × 9  (DW) = 71.7M
            + 2 × 256 × 56 × 56 × 128 (PW) = 102.8M
            = 174.5M FLOPs  →  5.3배 감소

MobileNetV1 vs ResNet-50 (ImageNet):
  MobileNetV1: 569M FLOPs, top-1 72.0%
  ResNet-50:  4100M FLOPs, top-1 76.1%
  → MobileNet은 단 4.1% 정확도 손실로 7.2배 저렴한 FLOPs

MobileNetV2 (역전 잔차):
  300M FLOPs, top-1 72.0% (MobileNetV1의 569M 대비)
  → 역전 잔차 + 선형 병목으로 연산 추가 감소
```

---

## 6. 깊이별 분리 합성곱을 사용할 때

```
하드웨어 고려사항:
  - DWS: ARM CPU(NEON SIMD), 모바일 NPU에 좋음
  - 표준: GPU에 좋음 (대규모 행렬 곱이 GPU에 더 효율적)
  - 깊이별만 단독 사용 시 산술 강도가 낮음 → GPU에서 메모리 병목

DWS 사용 시:
  - 모바일/엣지 하드웨어에 배포할 때
  - FLOPs 예산이 주요 제약일 때
  - 배치 크기가 작을 때 (추론)

표준 합성곱 선호 시:
  - cuDNN/cuBLAS GEMM이 고도로 최적화된 GPU 훈련
  - 정확도가 주요 목표일 때 (DWS는 표현 능력에서 약간 손실)
```

---

## 핵심 정리

- **깊이별 분리 합성곱** = 깊이별(채널당 공간 필터링) + 점별(1×1 채널 혼합)
- FLOP 감소: `1/C_out + 1/K²` — K=3, C_out≥64에서 약 8~9배
- 깊이별 역전파: 기울기가 채널 독립적으로 흐름 — dX, dW에서 채널 간 혼합 없음
- 점별 합성곱 = 행렬 곱 — NCHW→NHWC 재구성 후 `cblas_sgemm`으로 구현
- **MobileNetV2**는 최첨단 효율성을 위해 역전 잔차(확장 → DW → 투영) 사용

---

**다음**: [18. 스퀴즈-익사이테이션과 어텐션](./18_Squeeze_Excitation_and_Attention.md) — 채널 어텐션(SE 블록), 공간 어텐션(CBAM), 그리고 이것이 비전 트랜스포머를 어떻게 준비하는지.
