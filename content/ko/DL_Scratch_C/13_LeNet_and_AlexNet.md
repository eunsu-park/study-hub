# 13. LeNet과 AlexNet

**이전**: [이미지 데이터 파이프라인](./12_Data_Pipeline_Images.md) | **다음**: [CIFAR-10에서 CNN 학습하기](./14_Training_CNN_CIFAR10.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 이전 레슨에서 만든 conv, pool, activation 기본 요소를 사용하여 LeNet-5를 직접 구현하기
2. AlexNet이 2012년에 도입한 아키텍처 혁신 설명하기
3. AlexNet 스타일 레이어 구현 (LRN은 선택사항; conv + ReLU + pool 쌓기에 집중)
4. 각 레이어의 파라미터 수를 세고 메모리 요구사항 이해하기
5. 두 네트워크를 통한 forward pass를 구체적인 shape로 추적하기

---

## 1. LeNet-5 (LeCun, 1998)

### 아키텍처

```
Input: [N, 1, 32, 32]   (그레이스케일, MNIST의 28×28을 패딩)

C1:  Conv(1→6,  5×5, s=1, p=0)  → [N, 6, 28, 28]   + Tanh
S2:  AvgPool(2×2, s=2)           → [N, 6, 14, 14]
C3:  Conv(6→16, 5×5, s=1, p=0)  → [N, 16, 10, 10]  + Tanh
S4:  AvgPool(2×2, s=2)           → [N, 16, 5, 5]
C5:  Conv(16→120, 5×5, s=1, p=0)→ [N, 120, 1, 1]   + Tanh
F6:  FC(120 → 84)                + Tanh
OUT: FC(84 → 10)                 + Softmax
```

파라미터 수:

```
C1:  (5×5×1  + 1) × 6    =   156
C3:  (5×5×6  + 1) × 16   = 2,416
C5:  (5×5×16 + 1) × 120  = 48,120
F6:  (120 + 1) × 84       = 10,164
OUT: (84 + 1) × 10         =    850
합계:                      ~61,706 파라미터
```

### 구현

```c
// LeNet-5 forward pass
// Input: [N, 1, 32, 32]   Output: logits [N, 10]
void lenet5_forward(
    const float *X,          // [N, 1, 32, 32]
    float       *logits,     // [N, 10]
    LeNet5Weights *wt,       // 가중치 구조체
    LeNet5Buffers *buf,      // 중간 버퍼
    int N) {

    // C1: Conv(1→6, 5×5) + Tanh
    int OH1 = 28, OW1 = 28;
    conv2d_naive(X, N, 1, 32, 32,
                 wt->c1_w, 6, 5, 5,
                 buf->c1_out, OH1, OW1, 1, 0, 1);
    add_bias_chw(buf->c1_out, wt->c1_b, N, 6, OH1, OW1);
    apply_tanh(buf->c1_out, N * 6 * OH1 * OW1);

    // S2: AvgPool(2×2, s=2)
    avg_pool2d_forward(buf->c1_out, buf->s2_out,
                       N, 6, OH1, OW1, 2, 2, 14, 14, 2, 0);

    // C3: Conv(6→16, 5×5) + Tanh
    int OH3 = 10, OW3 = 10;
    conv2d_naive(buf->s2_out, N, 6, 14, 14,
                 wt->c3_w, 16, 5, 5,
                 buf->c3_out, OH3, OW3, 1, 0, 1);
    add_bias_chw(buf->c3_out, wt->c3_b, N, 16, OH3, OW3);
    apply_tanh(buf->c3_out, N * 16 * OH3 * OW3);

    // S4: AvgPool(2×2, s=2)
    avg_pool2d_forward(buf->c3_out, buf->s4_out,
                       N, 16, OH3, OW3, 2, 2, 5, 5, 2, 0);

    // C5: Conv(16→120, 5×5) → [N, 120, 1, 1]
    conv2d_naive(buf->s4_out, N, 16, 5, 5,
                 wt->c5_w, 120, 5, 5,
                 buf->c5_out, 1, 1, 1, 0, 1);
    add_bias_chw(buf->c5_out, wt->c5_b, N, 120, 1, 1);
    apply_tanh(buf->c5_out, N * 120);

    // F6: FC(120 → 84) + Tanh
    // GAP 후 buf->c5_out shape: [N, 120]
    matmul(buf->c5_out, wt->f6_w, buf->f6_out, N, 120, 84);
    add_bias_vec(buf->f6_out, wt->f6_b, N, 84);
    apply_tanh(buf->f6_out, N * 84);

    // OUT: FC(84 → 10)
    matmul(buf->f6_out, wt->out_w, logits, N, 84, 10);
    add_bias_vec(logits, wt->out_b, N, 10);
    // Softmax는 loss 계산 시 적용
}

// 보조 함수: CHW 출력에 bias 추가 (H,W에 대해 브로드캐스트)
void add_bias_chw(float *X, const float *b, int N, int C, int H, int W) {
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++) {
        float bv = b[c];
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            NCHW(X, N, C, H, W, n, c, h, w) += bv;
    }
}
```

---

## 2. AlexNet (Krizhevsky et al., 2012)

LeNet 대비 AlexNet의 주요 혁신:

```
1. ReLU 활성화 (tanh 대신) — 그래디언트 소실 방지
2. MaxPool (average pool 대신) — 더 선명한 특징 선택
3. Dropout (p=0.5) in FC 레이어 — 과적합 감소
4. 데이터 증강 (crop, flip, color jitter)
5. 두 개의 GTX 580에서 GPU 학습 — GPU DL 확장성 입증
```

### 아키텍처 (CIFAR-10용, 32×32 입력으로 적용)

원본 AlexNet은 ImageNet 224×224용으로 설계됨. CIFAR-10에서는 첫 번째 레이어들을 조정:

```
Input: [N, 3, 32, 32]

L1: Conv(3→64,   3×3, s=1, p=1)  → [N, 64, 32, 32]  + ReLU
L2: Conv(64→192, 3×3, s=1, p=1)  → [N, 192, 32, 32] + ReLU + MaxPool(2×2,s=2)
    → [N, 192, 16, 16]
L3: Conv(192→384,3×3, s=1, p=1)  → [N, 384, 16, 16] + ReLU
L4: Conv(384→256,3×3, s=1, p=1)  → [N, 256, 16, 16] + ReLU
L5: Conv(256→256,3×3, s=1, p=1)  → [N, 256, 16, 16] + ReLU + MaxPool(2×2,s=2)
    → [N, 256, 8, 8]
GAP: GlobalAvgPool                 → [N, 256]
FC1: FC(256 → 256) + ReLU + Dropout(0.5)
FC2: FC(256 → 10)
```

파라미터 수:

```
L1: (3×3×3   + 1) × 64  =   1,792
L2: (3×3×64  + 1) × 192 = 110,784
L3: (3×3×192 + 1) × 384 = 663,936
L4: (3×3×384 + 1) × 256 = 884,992
L5: (3×3×256 + 1) × 256 = 590,080
FC1: (256 + 1) × 256    =  65,792
FC2: (256 + 1) × 10     =   2,570
합계:                    ~230만 파라미터
```

### Dropout

```c
// dropout_forward: inverted dropout 적용 (학습 시 스케일링)
// backward pass를 위해 mask 저장
void dropout_forward(
    float   *X,    // in-place
    uint8_t *mask, // [size] — 1 = 유지, 0 = 드롭
    int size, float p,   // 드롭 확률
    int training) {

    if (!training) return;  // eval 시 dropout 없음

    for (int i = 0; i < size; i++) {
        mask[i] = ((float)rand() / RAND_MAX) > p ? 1 : 0;
        X[i] *= mask[i] / (1.0f - p);  // inverted dropout 스케일링
    }
}

// dropout_backward: 드롭된 유닛의 그래디언트를 0으로 만들기
void dropout_backward(
    float         *dX,
    const uint8_t *mask,
    int size, float p) {

    for (int i = 0; i < size; i++)
        dX[i] *= mask[i] / (1.0f - p);
}
```

### ReLU

```c
// relu_forward: in-place
void relu_forward(float *X, int size) {
    for (int i = 0; i < size; i++)
        X[i] = X[i] > 0.0f ? X[i] : 0.0f;
}

// relu_backward: X > 0인 곳에서만 그래디언트 통과
// 여기서 X = ReLU 이전 활성화 또는 동일하게 output > 0
void relu_backward(float *dX, const float *Y, int size) {
    for (int i = 0; i < size; i++)
        dX[i] *= (Y[i] > 0.0f) ? 1.0f : 0.0f;
}
```

---

## 3. 가중치 초기화

적절한 초기화는 그래디언트 소실/폭발을 방지합니다:

```c
#include <math.h>

// He 초기화: ReLU 네트워크에 적합
// std = sqrt(2 / fan_in)
void he_init(float *W, int fan_in, int fan_out) {
    float std = sqrtf(2.0f / fan_in);
    for (int i = 0; i < fan_in * fan_out; i++)
        W[i] = randn() * std;
}

// Xavier/Glorot 초기화: tanh/sigmoid 네트워크에 적합
// std = sqrt(2 / (fan_in + fan_out))
void xavier_init(float *W, int fan_in, int fan_out) {
    float std = sqrtf(2.0f / (fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++)
        W[i] = randn() * std;
}

// 표준 정규 샘플을 위한 Box-Muller 변환
float randn(void) {
    float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 1);
    float u2 = (float)rand()       / ((float)RAND_MAX + 1);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

// LeNet-5 초기화 (Xavier, tanh 네트워크)
void lenet5_init_weights(LeNet5Weights *wt) {
    xavier_init(wt->c1_w, 1 * 5 * 5, 6);
    xavier_init(wt->c3_w, 6 * 5 * 5, 16);
    xavier_init(wt->c5_w, 16 * 5 * 5, 120);
    xavier_init(wt->f6_w, 120, 84);
    xavier_init(wt->out_w, 84, 10);
    // bias는 0으로
}
```

---

## 4. 아키텍처 비교

```
          LeNet-5          AlexNet (적용됨)
Input     1×32×32          3×32×32
파라미터  ~62K             ~2.3M
활성화    Tanh             ReLU
풀링      AvgPool          MaxPool
Dropout   없음             있음 (FC 레이어)
BN        없음             없음 (BN 이전)
증강      없음             Flip + Crop + Jitter

CIFAR-10 정확도 (근사):
  LeNet-5:         ~68%
  AlexNet (소형):  ~85%
  ResNet-20:       ~92%
```

---

## 5. 온전성 검사: Shape 추적

```c
static void shape_trace_lenet5(void) {
    int N = 2;
    // [2, 1, 32, 32] → C1 → [2, 6, 28, 28]
    printf("After C1:  [%d, 6, 28, 28]\n", N);
    // → S2 → [2, 6, 14, 14]
    printf("After S2:  [%d, 6, 14, 14]\n", N);
    // → C3 → [2, 16, 10, 10]
    printf("After C3:  [%d, 16, 10, 10]\n", N);
    // → S4 → [2, 16, 5, 5]
    printf("After S4:  [%d, 16, 5, 5]\n", N);
    // → C5 → [2, 120, 1, 1]
    printf("After C5:  [%d, 120, 1, 1]\n", N);
    // → F6 → [2, 84]
    printf("After F6:  [%d, 84]\n", N);
    // → OUT → [2, 10]
    printf("After OUT: [%d, 10]\n", N);
}
```

---

## 핵심 정리

- **LeNet-5**: 62K 파라미터, tanh + avg-pool, MNIST/소형 그레이스케일 이미지에 적합
- **AlexNet**: 230만 파라미터, ReLU + max-pool + dropout — 2012년 ImageNet 돌파구
- **He 초기화** (`std = sqrt(2/fan_in)`)는 ReLU 네트워크에 적합한 선택; tanh에는 **Xavier**
- 학습 중 Dropout, eval 중 우회 — backward에서 mask를 저장하고 사용하는 것을 기억할 것
- 두 네트워크 모두 이전 레슨들의 동일한 기본 요소를 재사용: conv, pool, activation, matmul

---

**다음**: [14. CIFAR-10에서 CNN 학습하기](./14_Training_CNN_CIFAR10.md) — 엔드-투-엔드 학습: 데이터 로더 + forward pass + cross-entropy loss + backward + SGD 옵티마이저 + 정확도 측정.
