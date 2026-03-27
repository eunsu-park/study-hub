# 16. ResNet과 스킵 연결

**이전**: [VGG와 깊은 신경망](./15_VGG_and_Deep_Networks.md) | **다음**: [깊이별 분리 합성곱](./17_Depthwise_Separable_Conv.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 스킵 연결(skip connection)이 기울기 소실 문제를 해결하는 이유 설명하기
2. C 언어로 항등 지름길(identity shortcut)과 투영 지름길(projection shortcut)을 포함한 잔차 블록 구현하기
3. 이전 단원의 구성 요소를 사용해 CIFAR-10을 위한 ResNet-20 구축하기
4. 잔차 연결을 통한 역전파(기울기 분기) 유도하기
5. ResNet-20과 VGG 스타일 네트워크의 깊이 대 정확도 비교하기

---

## 1. 성능 저하 문제

He et al. (2015)은 직관에 반하는 현상을 관찰했습니다:

```
평범한 네트워크에 레이어를 추가하면 훈련 정확도가 오히려 나빠집니다:
  Plain-20:  CIFAR-10 오류율 8.82%
  Plain-56: CIFAR-10 오류율 13.47%   ← 더 깊을수록 더 나쁨!

이는 과적합이 아닙니다 (훈련 오류도 나빠짐).
네트워크가 여러 레이어를 통해 항등 변환을 학습하지 못하는 것입니다.
```

해결책: 네트워크가 완전한 변환이 아닌 **잔차(residual)** (보정값)를 학습하도록 가르치기.

---

## 2. 잔차 블록

`H(x)`를 학습하는 대신, `F(x) = H(x) - x` (잔차)를 학습합니다:

```
순전파:
  y = F(x, {W}) + x
  여기서 F는 2~3개의 conv+BN+ReLU 레이어를 쌓은 것

항등 매핑의 경우: F(x) = 0 학습 (쉬움 — 가중치를 0으로)
보정의 경우:      작은 F(x) = Δ 학습

역전파:
  ∂L/∂x = ∂L/∂y × (∂F/∂x + I)
                       ↑ 항등 행렬
         = ∂L/∂y × ∂F/∂x  +  ∂L/∂y

기울기는 지름길 경로를 통해 항상 최소한 ∂L/∂y를 받습니다.
100개 이상의 레이어에서도 기울기 소실이 없습니다.
```

---

## 3. ResNet 구성 요소

### 기본 블록 (ResNet-18/34/20)

두 개의 3×3 합성곱:

```
x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+x) → ReLU → 출력
↓___________________________________________↑  (스킵)
```

```c
typedef struct {
    // 주 경로: conv1 → bn1 → relu → conv2 → bn2
    float *conv1_w, *conv1_b;   // [C_out, C_in, 3, 3]
    BatchNorm *bn1;
    float *conv2_w, *conv2_b;   // [C_out, C_out, 3, 3]
    BatchNorm *bn2;
    // 투영 지름길 (stride>1 또는 C_in != C_out일 때)
    float *proj_w, *proj_b;     // [C_out, C_in, 1, 1]  (항등이면 NULL)
    BatchNorm *proj_bn;         // 투영에 적용할 BN
    int C_in, C_out, stride;
} ResidualBlock;

// 순전파
void resblock_forward(
    ResidualBlock *blk,
    const float   *X,      // [N, C_in, H, W]
    float         *Y,      // [N, C_out, OH, OW]
    float         *buf1,   // [N, C_out, OH, OW] — 중간값
    float         *buf_sc, // [N, C_out, OH, OW] — 지름길 분기
    float         *xhat1, *xhat2, *xhat_sc,  // BN 저장 상태
    int N, int H, int W,
    int training) {

    int OH = conv_output_size(H, 3, blk->stride, 1, 1);
    int OW = conv_output_size(W, 3, blk->stride, 1, 1);

    // 주 경로: conv1 → BN → ReLU
    conv2d_im2col(X, N, blk->C_in, H, W,
                  blk->conv1_w, blk->C_out, 3, 3,
                  buf1, OH, OW, blk->stride, 1, 1);
    add_bias_chw(buf1, blk->conv1_b, N, blk->C_out, OH, OW);
    // BN
    bn_forward_train(buf1, blk->bn1->gamma, blk->bn1->beta, buf1,
                     blk->bn1->mean, blk->bn1->var, xhat1,
                     blk->bn1->run_mean, blk->bn1->run_var,
                     0.1f, N, blk->C_out, OH, OW);
    relu_forward(buf1, N * blk->C_out * OH * OW);

    // 주 경로: conv2 → BN (아직 ReLU 없음 — 먼저 지름길 추가)
    conv2d_im2col(buf1, N, blk->C_out, OH, OW,
                  blk->conv2_w, blk->C_out, 3, 3,
                  Y, OH, OW, 1, 1, 1);
    add_bias_chw(Y, blk->conv2_b, N, blk->C_out, OH, OW);
    bn_forward_train(Y, blk->bn2->gamma, blk->bn2->beta, Y,
                     blk->bn2->mean, blk->bn2->var, xhat2,
                     blk->bn2->run_mean, blk->bn2->run_var,
                     0.1f, N, blk->C_out, OH, OW);

    // 지름길 분기
    if (blk->proj_w) {
        // 투영: 1×1 conv + BN (stride > 1 또는 채널 불일치 시)
        conv2d_im2col(X, N, blk->C_in, H, W,
                      blk->proj_w, blk->C_out, 1, 1,
                      buf_sc, OH, OW, blk->stride, 0, 1);
        add_bias_chw(buf_sc, blk->proj_b, N, blk->C_out, OH, OW);
        bn_forward_train(buf_sc, blk->proj_bn->gamma, blk->proj_bn->beta, buf_sc,
                         blk->proj_bn->mean, blk->proj_bn->var, xhat_sc,
                         blk->proj_bn->run_mean, blk->proj_bn->run_var,
                         0.1f, N, blk->C_out, OH, OW);
    } else {
        // 항등 지름길: X를 직접 복사
        memcpy(buf_sc, X, N * blk->C_out * OH * OW * sizeof(float));
    }

    // 지름길을 주 경로에 더하고 ReLU 적용
    int sz = N * blk->C_out * OH * OW;
    for (int i = 0; i < sz; i++) Y[i] += buf_sc[i];
    relu_forward(Y, sz);
}
```

---

## 4. 스킵 연결을 통한 역전파

기울기는 덧셈 노드에서 분기됩니다:

```
∂L/∂x = ∂L/∂y × ∂y/∂x
       = ∂L/∂y × (∂F/∂x + ∂shortcut/∂x)

항등 지름길의 경우 (∂shortcut/∂x = I):
  dX_total = dX_from_main_path + dY   (스킵 경로의 기울기가 변환 없이 통과)

투영 지름길의 경우:
  dX_total = dX_from_main_path + dX_from_projection_conv
```

```c
// resblock_backward: dY에서 dX 계산
void resblock_backward(
    ResidualBlock *blk,
    const float   *X, const float *buf1,  // 순전파에서 저장
    const float   *dY,
    float         *dX,    // [N, C_in, H, W]
    float         *dW1, float *db1,
    float         *dW2, float *db2,
    float         *dWp, float *dbp,   // 투영 (있는 경우)
    float         *dgamma1, *dbeta1,
    float         *dgamma2, *dbeta2,
    float         *dgammap, *dbetap,
    const float   *xhat1, *xhat2, *xhat_sc,
    int N, int H, int W) {

    int OH = conv_output_size(H, 3, blk->stride, 1, 1);
    int OW = conv_output_size(W, 3, blk->stride, 1, 1);

    // ---- 지름길 기울기 ----
    float *dX_skip = calloc(N * blk->C_in * H * W, sizeof(float));
    if (blk->proj_w) {
        // dY → BN 역전파 → 1×1 conv 역전파
        float *dY_proj = malloc(N * blk->C_out * OH * OW * sizeof(float));
        memcpy(dY_proj, dY, N * blk->C_out * OH * OW * sizeof(float));
        float *d_proj_in = malloc(N * blk->C_out * OH * OW * sizeof(float));
        bn_backward(dY_proj, xhat_sc, blk->proj_bn->gamma, blk->proj_bn->var,
                    d_proj_in, dgammap, dbetap, N, blk->C_out, OH, OW);
        float *dcol = malloc(N * OH * OW * blk->C_in * 1 * 1 * sizeof(float));
        input_backward(d_proj_in, blk->proj_w, dcol, dX_skip,
                       N, blk->C_in, H, W, blk->C_out, 1, 1, OH, OW,
                       blk->stride, 0, 1);
        // 가중치 기울기
        float *col = malloc(N * OH * OW * blk->C_in * sizeof(float));
        im2col(X, N, blk->C_in, H, W, 1, 1, OH, OW, blk->stride, 0, 1, col);
        weight_backward(col, d_proj_in, dWp, N * OH * OW, blk->C_in, blk->C_out);
        free(dY_proj); free(d_proj_in); free(dcol); free(col);
    } else {
        // 항등: 스킵 기울기는 단순히 dY
        memcpy(dX_skip, dY, N * blk->C_in * H * W * sizeof(float));
    }

    // ---- 주 경로 역전파 (BN2 → Conv2 → ReLU → BN1 → Conv1) ----
    float *dY_main = malloc(N * blk->C_out * OH * OW * sizeof(float));
    memcpy(dY_main, dY, N * blk->C_out * OH * OW * sizeof(float));

    // BN2 역전파
    float *d_conv2_out = malloc(N * blk->C_out * OH * OW * sizeof(float));
    bn_backward(dY_main, xhat2, blk->bn2->gamma, blk->bn2->var,
                d_conv2_out, dgamma2, dbeta2, N, blk->C_out, OH, OW);

    // Conv2 역전파
    float *d_relu1 = calloc(N * blk->C_out * OH * OW, sizeof(float));
    float *dcol2 = malloc(N * OH * OW * blk->C_out * 9 * sizeof(float));
    input_backward(d_conv2_out, blk->conv2_w, dcol2, d_relu1,
                   N, blk->C_out, OH, OW, blk->C_out, 3, 3, OH, OW, 1, 1, 1);
    float *col2 = malloc(N * OH * OW * blk->C_out * 9 * sizeof(float));
    im2col(buf1, N, blk->C_out, OH, OW, 3, 3, OH, OW, 1, 1, 1, col2);
    weight_backward(col2, d_conv2_out, dW2, N * OH * OW, blk->C_out * 9, blk->C_out);
    free(dcol2); free(col2); free(dY_main); free(d_conv2_out);

    // ReLU 역전파 (buf1 = relu_forward 후 ReLU 이전 출력, Y>0 마스크로 저장)
    relu_backward(d_relu1, buf1, N * blk->C_out * OH * OW);

    // BN1 역전파
    float *d_conv1_out = malloc(N * blk->C_out * OH * OW * sizeof(float));
    bn_backward(d_relu1, xhat1, blk->bn1->gamma, blk->bn1->var,
                d_conv1_out, dgamma1, dbeta1, N, blk->C_out, OH, OW);
    free(d_relu1);

    // Conv1 역전파 → dX_main 생성
    float *dX_main = calloc(N * blk->C_in * H * W, sizeof(float));
    float *dcol1 = malloc(N * OH * OW * blk->C_in * 9 * sizeof(float));
    input_backward(d_conv1_out, blk->conv1_w, dcol1, dX_main,
                   N, blk->C_in, H, W, blk->C_out, 3, 3, OH, OW,
                   blk->stride, 1, 1);
    float *col1 = malloc(N * OH * OW * blk->C_in * 9 * sizeof(float));
    im2col(X, N, blk->C_in, H, W, 3, 3, OH, OW, blk->stride, 1, 1, col1);
    weight_backward(col1, d_conv1_out, dW1, N * OH * OW, blk->C_in * 9, blk->C_out);
    free(d_conv1_out); free(dcol1); free(col1);

    // 총 dX = 주 경로 + 스킵
    int in_sz = N * blk->C_in * H * W;
    for (int i = 0; i < in_sz; i++) dX[i] = dX_main[i] + dX_skip[i];
    free(dX_main); free(dX_skip);
}
```

---

## 5. CIFAR-10을 위한 ResNet-20

He et al. (2016)은 CIFAR 전용 ResNet을 제안했습니다:

```
입력: [N, 3, 32, 32]

스템: Conv(3→16, 3×3, p=1) → BN → ReLU    [N, 16, 32, 32]

스테이지 1: 3 × ResBlock(16→16,  s=1)         [N, 16, 32, 32]
스테이지 2: 3 × ResBlock(16→32,  첫 s=2)   [N, 32, 16, 16]
스테이지 3: 3 × ResBlock(32→64,  첫 s=2)   [N, 64, 8, 8]

GAP:  [N, 64]
FC:   64 → 10

파라미터:
  스템:           (3×3×3+1)×16    = 448
  스테이지 1: 3×[ (3×3×16+1)×16×2 ] = 3×4,640 = 13,920
  스테이지 2: 3×[ (3×3×32+1)×32×2 ] = 3×18,496 = 55,488
    + 투영: (1×1×16+1)×32  = 544
  스테이지 3: 3×[ (3×3×64+1)×64×2 ] = 3×73,856 = 221,568
    + 투영: (1×1×32+1)×64  = 2,112
  FC:  (64+1)×10                 = 650
  합계:                          ≈ 27만 파라미터

성능:
  ResNet-20:  CIFAR-10에서 91.25%
  ResNet-56:  CIFAR-10에서 93.03%
  VGG-16:     ~93%   (하지만 파라미터는 522배 많음!)
```

---

## 6. 항등 vs 투영 지름길

각각을 사용하는 경우:

```
항등 지름길 (X → 출력으로 직접):
  조건: stride=1 AND C_in == C_out
  비용: 추가 파라미터 없음

투영 지름길 (1×1 conv + BN):
  조건: stride > 1 OR C_in != C_out
  비용: C_in × C_out × 1×1 파라미터
  목적: 공간 크기(stride)와 채널 수 맞추기

ResNet 옵션 A: 항등에 0으로 패딩 (추가 파라미터 없음)
ResNet 옵션 B: 다운샘플링에만 투영  ← 원래 논문 (모범 사례)
ResNet 옵션 C: 모든 지름길을 투영으로
```

---

## 7. 기울기 흐름: 평범한 네트워크 vs ResNet

```
50레이어 평범한 네트워크:
  레이어 1의 기울기 ≈ 50개 야코비안의 곱
  크기: (0.9)^50 ≈ 0.005 — 출력보다 200배 작음

50레이어 ResNet:
  ∂L/∂x_0 = Σ_k ∂L/∂x_k   (손실까지 모든 스킵 경로의 합)
  적어도 하나의 직접 경로가 항상 전체 기울기 크기를 전달
  → 지수적 감쇠 없음
```

---

## 핵심 정리

- **잔차 블록**: `y = F(x) + x` — 네트워크는 완전한 매핑 H가 아닌 보정값 F를 학습
- **스킵 연결 역전파**: `+` 노드에서 기울기가 분기되고, 스킵 경로는 모든 레이어를 통해 `∂L/∂y`를 변환 없이 전달 → 기울기 소실 제거
- **투영 지름길**: 차원이 변경될 때(stride 또는 채널) 1×1 conv 사용, 그 외에는 항등
- **ResNet-20**은 VGG의 1억 3,800만 파라미터 대비 단 27만 파라미터로 CIFAR-10에서 91.25% 달성 — 512배 효율적
- 핵심 통찰: 항등을 기본값으로 만들기(잔차 0 = 항등)는 합성곱 스택으로 항등을 학습하는 것보다 최적화하기 쉬움

---

**다음**: [17. 깊이별 분리 합성곱](./17_Depthwise_Separable_Conv.md) — 표준 합성곱을 깊이별(depthwise) + 점별(pointwise) 단계로 분해하여 ~8배 FLOP 절감 달성 — MobileNet의 기반.
