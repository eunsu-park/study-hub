# 11. 배치 정규화

**이전**: [풀링 레이어](./10_Pooling_Layers.md) | **다음**: [이미지 데이터 파이프라인](./12_Data_Pipeline_Images.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 학습 모드와 추론(eval) 모드 모두에서 BN forward pass 구현하기
2. 추론을 위한 이동 평균(running mean)과 분산(variance) 유지하기
3. 평균과 분산 계산을 통한 BN backward pass 유도 및 구현하기
4. 학습 가능한 파라미터인 gamma(scale)와 beta(shift) 적용하기
5. BN이 학습을 가속화하는 이유와 ReLU에 대한 상대적 위치 설명하기

---

## 1. Batch Normalization(배치 정규화) 수식

batch와 공간 차원에 걸친 활성화의 미니배치가 주어졌을 때:

```
Input:  X  [N, C, H, W]

각 채널 c에 대해:
  mean[c]    = (1/M) Σ X[n,c,h,w]       여기서 M = N*H*W
  var[c]     = (1/M) Σ (X[n,c,h,w] - mean[c])²
  X_hat[c]   = (X - mean[c]) / sqrt(var[c] + ε)
  Y          = gamma[c] * X_hat + beta[c]
```

`gamma`와 `beta`는 학습 가능한 파라미터입니다 (각각 1과 0으로 초기화).

---

## 2. 학습 모드 Forward Pass

```c
#define BN_EPS 1e-5f

// bn_forward_train: 채널별로 (N, H, W)에 대해 정규화
// backward pass를 위해 mean, var, X_hat 저장
void bn_forward_train(
    const float *X,       // [N, C, H, W]
    const float *gamma,   // [C]
    const float *beta,    // [C]
    float       *Y,       // [N, C, H, W]
    float       *mean,    // [C] — backward를 위해 저장
    float       *var,     // [C] — backward를 위해 저장
    float       *X_hat,   // [N, C, H, W] — backward를 위해 저장
    float       *run_mean, // [C] — in-place 업데이트 (EMA)
    float       *run_var,  // [C] — in-place 업데이트 (EMA)
    float       momentum,  // 보통 0.1
    int N, int C, int H, int W) {

    int M = N * H * W;  // 채널당 원소 수

    for (int c = 0; c < C; c++) {
        // 배치 평균 계산
        float m = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            m += NCHW(X, N, C, H, W, n, c, h, w);
        m /= M;
        mean[c] = m;

        // 배치 분산 계산
        float v = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float diff = NCHW(X, N, C, H, W, n, c, h, w) - m;
            v += diff * diff;
        }
        v /= M;
        var[c] = v;

        float inv_std = 1.0f / sqrtf(v + BN_EPS);

        // 정규화 및 스케일링
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float x_norm = (NCHW(X, N, C, H, W, n, c, h, w) - m) * inv_std;
            NCHW(X_hat, N, C, H, W, n, c, h, w) = x_norm;
            NCHW(Y,     N, C, H, W, n, c, h, w) = gamma[c] * x_norm + beta[c];
        }

        // 이동 통계 업데이트 (지수 이동 평균)
        run_mean[c] = (1.0f - momentum) * run_mean[c] + momentum * m;
        run_var[c]  = (1.0f - momentum) * run_var[c]  + momentum * v;
    }
}
```

---

## 3. 추론(Eval) 모드 Forward Pass

추론 시에는 고정된 이동 통계를 사용합니다 — 배치 의존성 없음:

```c
// bn_forward_eval: 저장된 이동 mean/var 사용 (무작위 배치 의존성 없음)
void bn_forward_eval(
    const float *X,
    const float *gamma,
    const float *beta,
    float       *Y,
    const float *run_mean,  // [C] — 고정됨
    const float *run_var,   // [C] — 고정됨
    int N, int C, int H, int W) {

    for (int c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(run_var[c] + BN_EPS);
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float x_norm = (NCHW(X, N, C, H, W, n, c, h, w) - run_mean[c]) * inv_std;
            NCHW(Y, N, C, H, W, n, c, h, w) = gamma[c] * x_norm + beta[c];
        }
    }
}
```

**학습 vs Eval 모드**:

```
학습:  배치별로 mean/var 계산 → 잡음이 있는 정규화 (일반화에 도움)
Eval:  EMA의 mean/var 사용 → 결정론적 출력 (배포 시 필수)

버그: eval 모드로 전환하지 않으면 테스트 분산이 커짐 (forward 호출마다 BN 통계가 변함)
```

---

## 4. Backward Pass 유도

BN backward는 평균과 분산이 X의 함수이기 때문에 가장 까다로운 부분입니다.

정의:
```
M      = N*H*W           (배치 공간 크기)
σ      = sqrt(var + ε)   (표준 편차)
x_hat  = (x - μ) / σ    (정규화된 입력, forward에서 저장)
Y      = γ * x_hat + β
```

그래디언트는 (μ와 σ²를 통한 표준 연쇄 법칙 유도):

```
dγ  = Σ (dY * x_hat)                            [C]
dβ  = Σ dY                                       [C]
dx_hat = dY * γ                                  [N,C,H,W]

dX = (1/M*σ) * [ M*dx_hat
                 - Σ(dx_hat)
                 - x_hat * Σ(dx_hat * x_hat) ]
```

구현:

```c
// bn_backward: dY로부터 dX, dgamma, dbeta 계산
void bn_backward(
    const float *dY,     // [N, C, H, W]
    const float *X_hat,  // [N, C, H, W] — forward에서 저장됨
    const float *gamma,  // [C]
    const float *var,    // [C] — 배치 분산, forward에서 저장됨
    float       *dX,     // [N, C, H, W]
    float       *dgamma, // [C]
    float       *dbeta,  // [C]
    int N, int C, int H, int W) {

    int M = N * H * W;

    for (int c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(var[c] + BN_EPS);

        // dγ와 dβ
        float dg = 0.0f, db_val = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float dy   = NCHW(dY,    N, C, H, W, n, c, h, w);
            float xhat = NCHW(X_hat, N, C, H, W, n, c, h, w);
            dg     += dy * xhat;
            db_val += dy;
        }
        dgamma[c] += dg;
        dbeta[c]  += db_val;

        // dx_hat = dY * gamma[c]
        // Sum1 = Σ dx_hat,  Sum2 = Σ (dx_hat * x_hat)
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float dxhat = NCHW(dY, N, C, H, W, n, c, h, w) * gamma[c];
            float xhat  = NCHW(X_hat, N, C, H, W, n, c, h, w);
            sum1 += dxhat;
            sum2 += dxhat * xhat;
        }

        // dX = (inv_std / M) * [M*dx_hat - sum1 - x_hat*sum2]
        float scale = inv_std / M;
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
            float dxhat = NCHW(dY,    N, C, H, W, n, c, h, w) * gamma[c];
            float xhat  = NCHW(X_hat, N, C, H, W, n, c, h, w);
            NCHW(dX, N, C, H, W, n, c, h, w) =
                scale * (M * dxhat - sum1 - xhat * sum2);
        }
    }
}
```

---

## 5. BN 레이어 구조체

모든 BN 상태를 캡슐화:

```c
typedef struct {
    int C;
    float *gamma, *beta;       // 학습 가능한 파라미터 [C]
    float *dgamma, *dbeta;     // 그래디언트 [C]
    float *run_mean, *run_var; // 이동 통계 [C]
    float *mean, *var;         // 배치 통계 [C] — backward를 위해 저장
    float *X_hat;              // 정규화된 입력 [N,C,H,W] — backward를 위해 저장
    float  momentum;           // EMA 감쇠율 (기본값 0.1)
    int    N, H, W;            // 마지막 forward 호출에서 저장됨
} BatchNorm;

BatchNorm *bn_create(int C, int N_max, int H_max, int W_max) {
    BatchNorm *bn = calloc(1, sizeof(BatchNorm));
    bn->C = C;
    bn->gamma    = malloc(C * sizeof(float));
    bn->beta     = calloc(C, sizeof(float));  // 0으로 초기화
    bn->dgamma   = calloc(C, sizeof(float));
    bn->dbeta    = calloc(C, sizeof(float));
    bn->run_mean = calloc(C, sizeof(float));
    bn->run_var  = malloc(C * sizeof(float));
    bn->mean     = malloc(C * sizeof(float));
    bn->var      = malloc(C * sizeof(float));
    bn->X_hat    = malloc(N_max * C * H_max * W_max * sizeof(float));
    bn->momentum = 0.1f;

    // gamma를 1.0으로 초기화
    for (int c = 0; c < C; c++) {
        bn->gamma[c]   = 1.0f;
        bn->run_var[c] = 1.0f;  // 첫 eval 호출 시 0으로 나누기 방지
    }
    return bn;
}

void bn_free(BatchNorm *bn) {
    free(bn->gamma); free(bn->beta);
    free(bn->dgamma); free(bn->dbeta);
    free(bn->run_mean); free(bn->run_var);
    free(bn->mean); free(bn->var); free(bn->X_hat);
    free(bn);
}
```

---

## 6. BN 배치 위치

현대 CNN에서의 표준 배치:

```
원본 (2015 논문): Conv → BN → ReLU
현대 관행:        Conv → BN → ReLU  (가장 일반적, ResNet에서 사용)
Pre-activation:   BN → ReLU → Conv  (ResNet-v2에서 사용)

참고: Transformer는 BN(배치 전체의 채널별)이 아닌
      LayerNorm(샘플별, 채널별)을 사용합니다.
      BN은 배치 크기 > 1이 필요; LN은 배치 크기 = 1에서도 작동.
```

---

## 7. BN이 효과적인 이유

```
BN 없이:
  - 파라미터 업데이트에 따라 레이어 출력이 큰 값으로 이동
  - 이후 레이어에서 분포 이동 발생 (internal covariate shift)
  - 신중한 초기화와 낮은 학습률 필요

BN 있이:
  - 각 레이어의 출력이 채널별로 ~N(0,1)로 재정규화됨
  - 파라미터 스케일에 관계없이 그래디언트가 잘 조건화됨
  - 정규화 효과 (미니배치 통계가 잡음 추가)
  - 10배 더 큰 학습률 허용 → 더 빠른 수렴
```

---

## 핵심 정리

- **학습 모드**: 배치별로 mean/var 계산; 이동 EMA 업데이트; backward를 위해 x_hat 저장
- **Eval 모드**: 고정된 이동 mean/var 사용 — 테스트 데이터로 통계를 계산하지 말 것
- **Backward**는 복잡합니다: 그래디언트가 gamma/beta 스케일링뿐만 아니라 정규화 자체(μ와 σ를 통해)를 통해 흐름
- `dgamma = Σ(dY * x_hat)` 및 `dbeta = Σ(dY)`는 간단합니다; `dX`는 배치 수준 제약을 설명하는 두 가지 보정 항을 포함
- BN은 학습을 극적으로 안정화 — 사실상 모든 현대 CNN에서 사용

---

**다음**: [12. 이미지 데이터 파이프라인](./12_Data_Pipeline_Images.md) — STB로 이미지 로딩, NHWC/NCHW 간 변환, 데이터 증강 (flip, crop, normalize).
