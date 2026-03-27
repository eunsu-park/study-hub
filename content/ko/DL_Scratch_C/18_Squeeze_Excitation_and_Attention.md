# 18. 스퀴즈-익사이테이션과 어텐션

**이전**: [깊이별 분리 합성곱](./17_Depthwise_Separable_Conv.md) | **다음**: [EfficientNet 스케일링](./19_EfficientNet_Scaling.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 채널 어텐션을 위한 스퀴즈-익사이테이션(Squeeze-Excitation, SE) 블록 구현하기
2. CBAM의 공간 어텐션 모듈 구현하기
3. SE 블록이 최소한의 파라미터 비용으로 정확도를 향상시키는 방법 설명하기
4. 채널 어텐션과 트랜스포머의 셀프 어텐션 메커니즘 연결하기
5. ResNet 병목 안에 SE 블록 적용하기 (SE-ResNet)

---

## 1. CNN에서 어텐션의 동기

표준 합성곱은 모든 채널을 동등하게 취급합니다 — 각 필터는 모든 입력 채널을 균일하게 혼합합니다. 실제로는 특정 입력에 대해 일부 채널이 다른 채널보다 더 유익합니다.

**채널 어텐션**이 답하는 질문: *이 입력을 주었을 때, 어떤 채널을 증폭해야 할까?*

```
표준 합성곱: Y = W × X  (고정 가중치, 채널 중요도 고정)
SE 블록:     Y = σ(FC(GAP(X))) ⊙ X  (입력당 동적 채널 스케일링)

⊙ = 요소별 곱셈 (공간 차원에 걸쳐 브로드캐스트)
σ = 시그모이드 (출력이 [0,1] = 어텐션 가중치)
```

---

## 2. 스퀴즈-익사이테이션 블록

Hu et al. (2018) — ILSVRC 2017 우승.

```
구조:
  X [N, C, H, W]
    ↓ 스퀴즈(Squeeze): GlobalAvgPool → [N, C]
    ↓ 익사이테이션(Excitation): FC(C → C/r) → ReLU → FC(C/r → C) → Sigmoid → [N, C]
    ↓ 스케일(Scale): 브로드캐스트 곱셈 → [N, C, H, W]
    = Y

  r = 감소 비율 (기본값 16): 파라미터 비용 제어
  FC 레이어는 작음: C×C/r + C/r×C = 2C²/r 파라미터
  C=256, r=16의 경우: 2×256²/16 = 8,192 파라미터  (합성곱: 256×256×9 = 589,824)
```

### 구현

```c
typedef struct {
    int C, C_reduced;     // C_reduced = C / reduction
    float *fc1_w;         // [C_reduced, C]
    float *fc1_b;         // [C_reduced]
    float *fc2_w;         // [C, C_reduced]
    float *fc2_b;         // [C]
} SEBlock;

SEBlock *se_create(int C, int reduction) {
    SEBlock *se = calloc(1, sizeof(SEBlock));
    se->C = C;
    se->C_reduced = C / reduction;
    se->fc1_w = malloc(se->C_reduced * C * sizeof(float));
    se->fc1_b = calloc(se->C_reduced, sizeof(float));
    se->fc2_w = malloc(C * se->C_reduced * sizeof(float));
    se->fc2_b = calloc(C, sizeof(float));
    // fc1, fc2에 He 초기화
    he_init(se->fc1_w, C, se->C_reduced);
    he_init(se->fc2_w, se->C_reduced, C);
    return se;
}

// 시그모이드 활성화
void sigmoid_forward(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = 1.0f / (1.0f + expf(-X[i]));
}

// se_forward: X에 SE 어텐션 적용, 결과를 Y에 저장 (Y == X이면 인플레이스)
void se_forward(
    SEBlock     *se,
    const float *X,     // [N, C, H, W]
    float       *Y,     // [N, C, H, W]
    float       *gap,   // [N, C] — GAP 출력
    float       *fc1,   // [N, C_reduced]
    float       *fc2,   // [N, C] — 어텐션 가중치
    int N, int C, int H, int W) {

    // 1. 스퀴즈: 전역 평균 풀링 [N,C,H,W] → [N,C]
    gap_forward(X, gap, N, C, H, W);

    // 2. 익사이테이션 FC1: [N,C] × [C,C_r]^T → [N,C_r] + 편향 → ReLU
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, se->C_reduced, C,
                1.0f, gap, C, se->fc1_w, C,
                0.0f, fc1, se->C_reduced);
    for (int n = 0; n < N; n++)
    for (int j = 0; j < se->C_reduced; j++)
        fc1[n * se->C_reduced + j] += se->fc1_b[j];
    relu_forward(fc1, N * se->C_reduced);

    // 3. 익사이테이션 FC2: [N,C_r] × [C_r,C]^T → [N,C] + 편향 → Sigmoid
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, C, se->C_reduced,
                1.0f, fc1, se->C_reduced, se->fc2_w, se->C_reduced,
                0.0f, fc2, C);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
        fc2[n * C + c] += se->fc2_b[c];
    sigmoid_forward(fc2, N * C);

    // 4. 스케일: 어텐션 가중치를 공간 차원에 걸쳐 브로드캐스트
    if (Y != X) memcpy(Y, X, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float scale = fc2[n * C + c];
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(Y, N, C, H, W, n, c, h, w) *= scale;
    }
}
```

---

## 3. SE-ResNet 블록

마지막 BN 이후, 지름길 덧셈 이전에 SE 삽입:

```c
// SE-ResNet 기본 블록 순전파
void se_resblock_forward(
    ResidualBlock *blk,
    SEBlock       *se,
    const float   *X,
    float         *Y,
    // ... 버퍼 ...
    int N, int H, int W, int training) {

    // 표준 ResBlock 순전파 (BN2까지, ReLU 없이)
    resblock_pre_relu_forward(blk, X, Y, ..., N, H, W, training);

    int OH = conv_output_size(H, 3, blk->stride, 1, 1);
    int OW = conv_output_size(W, 3, blk->stride, 1, 1);

    // 주 경로에 SE 어텐션 적용 (지름길 추가 전)
    float *gap = malloc(N * blk->C_out * sizeof(float));
    float *fc1 = malloc(N * (blk->C_out / 16) * sizeof(float));
    float *fc2 = malloc(N * blk->C_out * sizeof(float));
    se_forward(se, Y, Y, gap, fc1, fc2, N, blk->C_out, OH, OW);
    free(gap); free(fc1); free(fc2);

    // 지름길 추가, 이후 ReLU
    add_shortcut_and_relu(blk, X, Y, N, H, W, OH, OW);
}
```

**파라미터 오버헤드**:

```
SE 블록 r=16, C=256:
  FC1: 256 × 16 = 4,096
  FC2: 16 × 256 = 4,096
  합계: 8,192 파라미터

ResNet-50은 총 ~2,500만 파라미터
SE-ResNet-50은 ~250만 파라미터 추가 (10% 오버헤드)
ImageNet 정확도: ResNet-50 76.1% → SE-ResNet-50 77.6% (+1.5%)
```

---

## 4. 공간 어텐션 (CBAM)

CBAM(Woo et al., 2018)은 채널 + 공간 어텐션을 결합합니다:

```
채널 어텐션: "어떤 채널이 중요한가?"
  → SE 블록과 동일

공간 어텐션: "각 채널에서 어떤 위치가 중요한가?"
  입력:  [N, C, H, W]
  풀링:  채널 최대 풀링 + 채널 평균 풀링 → [N, 2, H, W]
  합성곱: 7×7 conv → [N, 1, H, W] → Sigmoid
  출력:  공간 어텐션 맵 [N, 1, H, W] × X
```

```c
// cbam_spatial_forward: 공간 어텐션 맵 계산
void cbam_spatial_forward(
    const float *X,         // [N, C, H, W]
    const float *sp_conv_w, // [1, 2, 7, 7] 공간 합성곱 필터
    float       *Y,         // [N, C, H, W] — 출력
    float       *pool_feats, // [N, 2, H, W] — 채널 풀링 결과
    float       *sp_attn,    // [N, 1, H, W] — 공간 어텐션 맵
    int N, int C, int H, int W) {

    // 채널 최대 풀링과 평균 풀링
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        float max_v = -FLT_MAX, avg_v = 0.0f;
        for (int c = 0; c < C; c++) {
            float v = NCHW(X, N, C, H, W, n, c, h, w);
            if (v > max_v) max_v = v;
            avg_v += v;
        }
        avg_v /= C;
        pool_feats[n * 2 * H * W + 0 * H * W + h * W + w] = avg_v;
        pool_feats[n * 2 * H * W + 1 * H * W + h * W + w] = max_v;
    }

    // [N, 2, H, W]에 7×7 conv → [N, 1, H, W]
    conv2d_naive(pool_feats, N, 2, H, W,
                 sp_conv_w, 1, 7, 7,
                 sp_attn, H, W, 1, 3, 1);  // pad=3 for same size

    // Sigmoid
    sigmoid_forward(sp_attn, N * H * W);

    // X에 어텐션 맵 곱하기
    if (Y != X) memcpy(Y, X, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++)
        NCHW(Y, N, C, H, W, n, c, h, w)
            *= sp_attn[n * H * W + h * W + w];
}
```

---

## 5. 트랜스포머 셀프 어텐션과의 연결

SE 블록은 트랜스포머 셀프 어텐션의 선구자입니다:

```
SE 블록 (채널 어텐션):
  q, k, v 모두 GAP(X)에서 옴       → 전역 컨텍스트
  어텐션 = σ(FC(FC(GAP(X))))      → 채널별 게이트
  출력 = 어텐션 ⊙ X             → 채널 재보정

트랜스포머 셀프 어텐션:
  Q = X × W_Q, K = X × W_K, V = X × W_V
  어텐션 = softmax(Q×K^T / √d)
  출력 = 어텐션 × V             → 위치 재보정

핵심 차이:
  SE: 채널에 걸친 어텐션 (C차원 게이트)
  셀프 어텐션: 시퀀스 위치에 걸친 어텐션 (N차원 가중치 행렬)
  ViT: 이미지 패치에 셀프 어텐션 적용 → 완전한 공간 재보정
```

---

## 6. 정확도 영향

```
ImageNet top-1 정확도:

ResNet-50:         76.1%
SE-ResNet-50:      77.6%  (+1.5%)  → 추가 파라미터 250만
CBAM-ResNet-50:    77.3%  (+1.2%)  → 추가 파라미터 10만 (공간 합성곱이 작음)

MobileNetV1:       70.9%
SE-MobileNetV1:    73.0%  (+2.1%)  ← 경량 네트워크에서 더 큰 이득

결론: SE 블록은 최고의 정확도/파라미터 트레이드오프를 가짐
      특히 소형/모바일 아키텍처에서 유익
```

---

## 핵심 정리

- **SE 블록**: GlobalAvgPool → FC → ReLU → FC → Sigmoid → 채널별 스케일; 10% 파라미터 오버헤드로 ~1.5% 정확도 향상
- **채널 어텐션**은 "이 입력에서 어떤 채널이 중요한가"를 동적으로 결정
- **CBAM**은 채널 어텐션 위에 공간 어텐션 추가 — "어디를 봐야 할지" 학습
- SE 블록은 트랜스포머 셀프 어텐션의 직접적인 전신; 둘 다 어텐션 가중치를 계산하고 곱셈으로 적용
- 잔차 블록의 마지막 conv/BN 이후, 지름길 덧셈 전에 SE 블록을 삽입

---

**다음**: [19. EfficientNet 스케일링](./19_EfficientNet_Scaling.md) — 깊이, 너비, 해상도의 복합 스케일링; EfficientNet-B0 아키텍처; NAS 탐색 공간 개념.
