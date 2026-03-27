# 19. EfficientNet 스케일링

**이전**: [스퀴즈-익사이테이션과 어텐션](./18_Squeeze_Excitation_and_Attention.md) | **다음**: [현대 CNN 벤치마크](./20_Modern_CNN_Benchmark.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 복합 스케일링 규칙과 단일 차원 스케일링보다 우수한 이유 설명하기
2. 기본 B0 아키텍처에서 EfficientNet-Bx 치수 계산하기
3. SE가 포함된 MBConv(모바일 역전 병목 합성곱) 블록 구현하기
4. MBConv 블록을 쌓아 EfficientNet-B0 구축하기
5. EfficientNet 변형의 FLOPs와 파라미터 수 프로파일링하기

---

## 1. 스케일링 문제

기존 네트워크들은 단일 차원으로 스케일링합니다:

```
깊이 스케일링 (ResNet):    레이어 추가
  ResNet-18 → 34 → 50 → 101 → 152
  기울기 소실로 정확도 정체

너비 스케일링 (WideResNet): 채널 수 증가
  WideResNet-28×10 (레이어당 10배 넓음)
  깊이 없이 정확도 포화

해상도 스케일링:            더 큰 입력 이미지 사용
  32×32 → 224×224 → 480×480
  더 많은 메모리와 연산 필요

문제: 하나의 차원을 스케일링하면 수확 체감에 부딪히고
     다른 차원이 병목이 됨
```

### 복합 스케일링 (Tan & Le, 2019)

고정 비율로 세 차원을 함께 스케일링:

```
복합 계수 φ가 주어졌을 때:
  깊이:     d = α^φ
  너비:     w = β^φ
  해상도:   r = γ^φ

조건: α × β² × γ² ≈ 2  (총 FLOP 스케일링 ≈ 2^φ)

EfficientNet의 경우: α=1.2, β=1.1, γ=1.15 (φ=1에서 NAS 그리드 탐색으로 발견)

EfficientNet-Bx 스케일링 표:
  B0: d=1.0, w=1.0, r=224  (기준선)
  B1: d=1.1, w=1.0, r=240
  B2: d=1.2, w=1.1, r=260
  B3: d=1.4, w=1.2, r=300
  B4: d=1.8, w=1.4, r=380
  B5: d=2.2, w=1.6, r=456
  B6: d=2.6, w=1.8, r=528
  B7: d=3.1, w=2.0, r=600
```

---

## 2. EfficientNet-B0 아키텍처

B0는 NAS로 찾은 기준선 — 7단계의 MBConv 블록:

```
스테이지  블록     #레이어  C_in→C_out  k  Stride  Expand
  0    MBConv1     1      32→16      3    1       1    (확장 없음)
  1    MBConv6     2      16→24      3    2       6
  2    MBConv6     2      24→40      5    2       6
  3    MBConv6     3      40→80      3    2       6
  4    MBConv6     3      80→112     5    1       6
  5    MBConv6     4      112→192    5    2       6
  6    MBConv6     1      192→320    3    1       6

스템:   Conv(3→32, 3×3, s=2)        [N,32,112,112]
스테이지 0-6:                          [N,320,7,7]
헤드:   Conv(320→1280, 1×1)          [N,1280,7,7]
GAP:                                 [N,1280]
Dropout(0.2)
FC:     1280→1000

총 파라미터: ~530만
ImageNet top-1: 77.1%  (ResNet-50: 2,560만 파라미터로 76.1% 대비)
```

---

## 3. SE가 포함된 MBConv 블록

```c
typedef struct {
    int C_in, C_mid, C_out, K, stride;
    // 확장 PW (expand=1이면 생략)
    float *pw1_w;    // [C_mid, C_in, 1, 1]
    BatchNorm *bn1;
    // 깊이별
    float *dw_w;     // [C_mid, K, K]
    float *dw_b;     // [C_mid]
    BatchNorm *bn2;
    // SE (EfficientNet에서 reduction=4)
    SEBlock *se;
    // 투영 PW
    float *pw2_w;    // [C_out, C_mid, 1, 1]
    BatchNorm *bn3;
    int skip;        // 항등 잔차가 적용되면 1
} MBConvBlock;

void mbconv_forward(
    MBConvBlock *blk,
    const float *X,    // [N, C_in, H, W]
    float       *Y,    // [N, C_out, OH, OW]
    int N, int H, int W, int training) {

    int OH = conv_output_size(H, blk->K, blk->stride, blk->K/2, 1);
    int OW = conv_output_size(W, blk->K, blk->stride, blk->K/2, 1);

    float *buf = malloc((size_t)N * blk->C_mid * H * W * sizeof(float));

    // 1. 확장 PW (expand=1, 즉 C_mid == C_in이면 건너뜀)
    if (blk->C_mid != blk->C_in) {
        pointwise_conv2d(X, blk->pw1_w, NULL, buf, N, blk->C_in, blk->C_mid, H, W);
        // BN + SiLU (Swish): x * sigmoid(x)
        float *xhat = malloc((size_t)N * blk->C_mid * H * W * sizeof(float));
        if (training)
            bn_forward_train(buf, blk->bn1->gamma, blk->bn1->beta, buf,
                             blk->bn1->mean, blk->bn1->var, xhat,
                             blk->bn1->run_mean, blk->bn1->run_var, 0.1f,
                             N, blk->C_mid, H, W);
        swish_forward(buf, N * blk->C_mid * H * W);
        free(xhat);
    } else {
        memcpy(buf, X, (size_t)N * blk->C_in * H * W * sizeof(float));
    }

    // 2. 깊이별 합성곱
    float *dw_out = malloc((size_t)N * blk->C_mid * OH * OW * sizeof(float));
    depthwise_conv2d_forward(buf, blk->dw_w, blk->dw_b, dw_out,
                             N, blk->C_mid, H, W,
                             blk->K, blk->K, OH, OW,
                             blk->stride, blk->K / 2);
    // BN + SiLU
    float *xhat2 = malloc((size_t)N * blk->C_mid * OH * OW * sizeof(float));
    if (training)
        bn_forward_train(dw_out, blk->bn2->gamma, blk->bn2->beta, dw_out,
                         blk->bn2->mean, blk->bn2->var, xhat2,
                         blk->bn2->run_mean, blk->bn2->run_var, 0.1f,
                         N, blk->C_mid, OH, OW);
    swish_forward(dw_out, N * blk->C_mid * OH * OW);
    free(xhat2); free(buf);

    // 3. SE 블록
    float *gap_buf = malloc(N * blk->C_mid * sizeof(float));
    float *fc1_buf = malloc(N * (blk->C_mid / 4) * sizeof(float));
    float *fc2_buf = malloc(N * blk->C_mid * sizeof(float));
    se_forward(blk->se, dw_out, dw_out, gap_buf, fc1_buf, fc2_buf,
               N, blk->C_mid, OH, OW);
    free(gap_buf); free(fc1_buf); free(fc2_buf);

    // 4. 투영 PW → BN (MBConv의 마지막 PW 이후 활성화 없음)
    pointwise_conv2d(dw_out, blk->pw2_w, NULL, Y, N, blk->C_mid, blk->C_out, OH, OW);
    float *xhat3 = malloc((size_t)N * blk->C_out * OH * OW * sizeof(float));
    if (training)
        bn_forward_train(Y, blk->bn3->gamma, blk->bn3->beta, Y,
                         blk->bn3->mean, blk->bn3->var, xhat3,
                         blk->bn3->run_mean, blk->bn3->run_var, 0.1f,
                         N, blk->C_out, OH, OW);
    free(xhat3); free(dw_out);

    // 5. 항등 잔차 (스킵 연결) — 차원이 일치할 때만
    if (blk->skip) {
        int sz = N * blk->C_out * OH * OW;
        for (int i = 0; i < sz; i++) Y[i] += X[i];
    }
}

// SiLU (Swish): x * sigmoid(x) — EfficientNet에서 사용하는 활성화 함수
void swish_forward(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = X[i] * (1.0f / (1.0f + expf(-X[i])));
}
```

---

## 4. 코드로 구현한 복합 스케일링

```c
typedef struct {
    float depth_mult;   // 스테이지당 레이어 수에 곱하는 배수
    float width_mult;   // 채널 수에 곱하는 배수
    int   resolution;   // 입력 이미지 크기
} EfficientNetConfig;

// 채널을 8의 가장 가까운 배수로 반올림 (하드웨어 정렬)
int round_channels(float c, float width_mult) {
    int ch = (int)(c * width_mult);
    int divisor = 8;
    int new_ch = ((ch + divisor/2) / divisor) * divisor;
    if (new_ch < (int)(0.9f * ch)) new_ch += divisor;
    return new_ch;
}

// 레이어 수(깊이)를 최소 1 이상으로 반올림
int round_depth(float d, float depth_mult) {
    return (int)ceilf(d * depth_mult);
}

static const EfficientNetConfig EFFICIENTNET_CONFIGS[] = {
    {1.0f, 1.0f, 224},  // B0
    {1.1f, 1.0f, 240},  // B1
    {1.2f, 1.1f, 260},  // B2
    {1.4f, 1.2f, 300},  // B3
    {1.8f, 1.4f, 380},  // B4
    {2.2f, 1.6f, 456},  // B5
    {2.6f, 1.8f, 528},  // B6
    {3.1f, 2.0f, 600},  // B7
};

// EfficientNet-Bx 모델 구축
void efficientnet_build(int b, int *total_params) {
    EfficientNetConfig cfg = EFFICIENTNET_CONFIGS[b];

    // 스테이지 채널 너비 (B0 기준선)
    int ch[] = {32, 16, 24, 40, 80, 112, 192, 320, 1280};
    int depths[] = {1, 2, 2, 3, 3, 4, 1};  // 스테이지별 반복 횟수

    printf("EfficientNet-B%d 설정 (resolution=%d):\n", b, cfg.resolution);
    *total_params = 0;
    for (int s = 0; s < 7; s++) {
        int Ci = round_channels(ch[s],   cfg.width_mult);
        int Co = round_channels(ch[s+1], cfg.width_mult);
        int d  = round_depth(depths[s],  cfg.depth_mult);
        printf("  스테이지 %d: C_in=%3d → C_out=%3d, 블록 %d개\n", s, Ci, Co, d);
    }
}
```

---

## 5. FLOP 및 파라미터 비교

```
네트워크         파라미터  FLOPs   ImageNet top-1
ResNet-50       25.6M   4.1B    76.1%
MobileNetV2     3.4M    300M    72.0%
EfficientNet-B0 5.3M    390M    77.1%  ← 5배 적은 파라미터로 ResNet-50 능가
EfficientNet-B1 7.8M    700M    79.1%
EfficientNet-B4 19.3M   4.2B    82.9%  ← ResNet-50과 동일한 FLOPs, +6.8% 정확도
EfficientNet-B7 66M     37B     84.3%

EfficientNet-B0 vs MobileNetV2 (유사한 FLOPs):
  MobileNetV2:  72.0%  (SE 없음, Swish 없음)
  B0:           77.1%  (+5.1% SE + Swish + NAS 아키텍처 덕분)
```

---

## 6. 확률적 깊이 (Drop Path)

EfficientNet은 훈련 중 확률적 깊이를 사용 — 전체 MBConv 블록을 무작위로 건너뜀:

```c
// drop_path: 배치에서 샘플을 무작위로 0으로 만듦 (샘플별, 블록별)
void drop_path_forward(float *X, int N, int C, int H, int W,
                       float drop_rate, int training) {
    if (!training || drop_rate == 0.0f) return;

    // 배치의 각 샘플을 독립적으로 유지하거나 제거
    for (int n = 0; n < N; n++) {
        if ((float)rand() / RAND_MAX < drop_rate) {
            // 이 샘플의 기여를 0으로 만듦 (블록 건너뜀)
            float *sample = X + (long)n * C * H * W;
            memset(sample, 0, (long)C * H * W * sizeof(float));
        }
    }
    // 살아남은 샘플을 스케일링하여 기대값 유지
    float scale = 1.0f / (1.0f - drop_rate);
    for (int i = 0; i < N * C * H * W; i++) X[i] *= scale;
}
```

---

## 핵심 정리

- **복합 스케일링**: 깊이 × 너비 × 해상도를 고정 비율로 함께 스케일링 — 단일 차원 스케일링보다 우수
- EfficientNet-B0은 NAS로 발견: SE 어텐션과 SiLU 활성화를 가진 7단계 MBConv 블록
- **MBConv**: 확장 → DW(K×K) → SE → 투영 — 깊이별 분리 합성곱과 채널 어텐션 결합
- EfficientNet-B0은 이전 모든 혁신을 결합하여 5배 적은 파라미터로 ResNet-50 정확도 달성
- SiLU/Swish 활성화 함수(`x × σ(x)`)는 깊은 네트워크에서 ReLU보다 ~0.5~1% 정확도 향상

---

**다음**: [20. 현대 CNN 벤치마크](./20_Modern_CNN_Benchmark.md) — CIFAR-10/100에서 LeNet, VGG, ResNet-20, EfficientNet-B0 비교: 정확도, 속도, 메모리 트레이드오프.
