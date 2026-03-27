# 15. VGG와 깊은 신경망

**이전**: [CIFAR-10에서 CNN 훈련하기](./14_Training_CNN_CIFAR10.md) | **다음**: [ResNet과 스킵 연결](./16_ResNet_and_Skip_Connections.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. VGG의 설계 철학(작은 필터를 깊게 쌓기) 설명하기
2. VGG-16의 파라미터 수를 계산하고 메모리가 어디에 쓰이는지 설명하기
3. 매우 깊은 네트워크에서 기울기 소실(vanishing gradient)이 발생하는 이유 증명하기
4. C 언어로 VGG 스타일 합성곱 블록 구현하기
5. 깊이에 따른 수용 필드(receptive field) 증가 측정하기

---

## 1. VGG 설계 철학

VGG(Simonyan & Zisserman, 2014)는 깊이 — 3×3 합성곱을 쌓아 올리는 방식 — 가 성능의 핵심 요소임을 보였습니다. 핵심 통찰:

```
두 개의 3×3 합성곱을 쌓으면 하나의 5×5와 동일한 수용 필드를 가집니다:
  수용 필드:  3×3 → 5×5 (2개 레이어 후) → 7×7 (3개 레이어 후)

하지만 2 × (3×3×C²) 파라미터 = 18C²  vs  하나의 5×5×C² = 25C²  (28% 파라미터 절감)
그리고 두 개의 3×3 레이어는 ReLU를 두 번 적용 → 더 많은 비선형성
```

### VGG-16 아키텍처

```
입력: [N, 3, 224, 224]

블록 1: Conv(3→64,3×3,p=1)×2  → MaxPool(2×2)  → [N, 64, 112, 112]
블록 2: Conv(64→128,3×3,p=1)×2 → MaxPool(2×2)  → [N, 128, 56, 56]
블록 3: Conv(128→256,3×3,p=1)×3 → MaxPool(2×2) → [N, 256, 28, 28]
블록 4: Conv(256→512,3×3,p=1)×3 → MaxPool(2×2) → [N, 512, 14, 14]
블록 5: Conv(512→512,3×3,p=1)×3 → MaxPool(2×2) → [N, 512, 7, 7]
평탄화: [N, 25088]
FC1:  25088 → 4096 + ReLU + Dropout(0.5)
FC2:  4096  → 4096 + ReLU + Dropout(0.5)
FC3:  4096  → 1000
```

파라미터 수:

```
블록 1: (3×3×3+1)×64 + (3×3×64+1)×64         =    1,792 +    36,928 =    38,720
블록 2: (3×3×64+1)×128 + (3×3×128+1)×128      =   73,856 +   147,584 =   221,440
블록 3: 3×(3×3×256+1)×256 + …                 =  590,080 + 1,180,160 + 1,180,160 = 2,950,400 (근사)
블록 4: 3×(3×3×512+1)×512                     =  2,359,808 × 3 = 7,079,424 (근사)
블록 5: 동일                                   =  7,079,424
FC1:   25088×4096 + 4096                        = 102,764,544
FC2:   4096×4096 + 4096                         =  16,781,312
FC3:   4096×1000 + 1000                         =   4,097,000
합계:                                            ≈ 1억 3,800만 파라미터

분류:
  합성곱 레이어:  ~1,470만   (전체의 11%)
  완전연결 레이어: ~1억 2,360만  (전체의 89%) ← 대부분의 파라미터가 FC에 있음!
```

**핵심 통찰**: VGG는 파라미터의 89%를 세 개의 완전연결(FC) 레이어에 씁니다. ResNet은 전역 평균 풀링(GAP)으로 이를 완전히 제거합니다.

---

## 2. VGG 블록 구현

```c
// VGG 합성곱 블록: Conv → BN → ReLU (`n_convs`번 반복)
typedef struct {
    int n_convs;
    // 각 서브 레이어의 합성곱 가중치
    float **conv_w;   // [n_convs] 각각 [C_out, C_in, 3, 3]
    float **conv_b;   // [n_convs] 각각 [C_out]
    // BN 파라미터 (현대 VGG는 BN 추가)
    BatchNorm **bn;   // [n_convs]
    int C_in, C_out;
} VGGBlock;

// VGGBlock 순전파
void vgg_block_forward(
    VGGBlock    *blk,
    const float *X,     // [N, C_in, H, W]
    float       *Y,     // [N, C_out, H, W]
    float       **bufs, // 중간 버퍼 [n_convs]
    int N, int H, int W,
    int training) {

    const float *cur_in = X;
    int C_cur = blk->C_in;

    for (int i = 0; i < blk->n_convs; i++) {
        float *cur_out = (i < blk->n_convs - 1) ? bufs[i] : Y;
        int C_out = blk->C_out;

        // Conv(3×3, pad=1) — H×W 크기 유지
        int OH = conv_output_size(H, 3, 1, 1, 1);  // 동일 크기
        int OW = conv_output_size(W, 3, 1, 1, 1);
        conv2d_im2col(cur_in, N, C_cur, H, W,
                      blk->conv_w[i], C_out, 3, 3,
                      cur_out, OH, OW, 1, 1, 1);
        add_bias_chw(cur_out, blk->conv_b[i], N, C_out, H, W);

        // BN
        float *xhat = malloc(N * C_out * H * W * sizeof(float));
        bn_forward_train(cur_out, blk->bn[i]->gamma, blk->bn[i]->beta,
                         cur_out, blk->bn[i]->mean, blk->bn[i]->var, xhat,
                         blk->bn[i]->run_mean, blk->bn[i]->run_var,
                         0.1f, N, C_out, H, W);
        free(xhat);

        // ReLU 인플레이스
        relu_forward(cur_out, N * C_out * H * W);

        cur_in = cur_out;
        C_cur  = C_out;
    }
}
```

---

## 3. 수용 필드 증가

동일 패딩의 3×3 합성곱과 stride=1에서 수용 필드(RF)는 선형적으로 증가합니다:

```
레이어 깊이:    1    2    3    4    5    6    7    8    9   10   13
RF (stride=1):  3    5    7    9   11   13   15   17   19   21   27

각 MaxPool(stride=2) 이후, 유효 RF는 두 배가 됩니다:
  블록 1 (합성곱 2개): RF = 5×5  → MaxPool 이후: 입력의 10×10 범위 커버
  블록 2 (합성곱 2개): 입력 공간에서 유효 RF = 14×14
  블록 3 (합성곱 3개): 원본 224×224 입력의 ~62×62 범위 커버
  블록 5: RF = ~212×212 ≈ 거의 전체 이미지 커버
```

RF 계산:

```c
int receptive_field(int n_layers, int kernel, int stride_per_layer) {
    int rf = 1;
    for (int i = 0; i < n_layers; i++)
        rf = rf + (kernel - 1) * stride_per_layer;
    return rf;
}
// n_convs=13 (VGG-16 합성곱 레이어), kernel=3, effective_stride=1 각각
// RF = 1 + 12*2 = 25 (풀링 stride는 별도로 추가)
```

---

## 4. 깊은 네트워크에서의 기울기 소실

스킵 연결 없이 깊이가 증가하면 역전파 기울기가 감소합니다:

```
L개 레이어의 tanh를 통한 기울기:
  ∂L/∂x_0 = ∏_{i=1}^{L} (∂x_i/∂x_{i-1}) = ∏ (W_i × tanh'(x_i))

tanh'(x) ≤ 1.0, 일반적인 가중치 ≈ 0.5 → 곱은 0.5^L로 감소

L=10:  0.5^10 ≈ 0.001   (기울기 1,000배 작아짐)
L=20:  0.5^20 ≈ 1e-6    (기울기 100만 배 작아짐)
```

**ReLU**는 이를 부분적으로 해결합니다 — 기울기가 0 또는 1 (활성 유닛에서 감소 없음):

```
∂ReLU/∂x = 1 if x > 0
            0 if x ≤ 0  ← 죽은 뉴런(dead neuron) 문제
```

**하지만 ReLU가 문제를 완전히 해결하지는 못합니다** — 매우 깊은 네트워크(20개 레이어 이상)는 여전히 성능이 저하됩니다.

```
ImageNet에서 VGG-16 테스트 정확도:
  VGG-11 (합성곱 8개): top-1 70.4%
  VGG-13 (합성곱 10개): top-1 71.3%
  VGG-16 (합성곱 13개): top-1 74.4%  ← 최적점
  VGG-19 (합성곱 16개): top-1 74.5%  ← 거의 개선 없음! 깊이 한계 도달
  ResNet-50:            top-1 76.1%   ← 스킵 연결로 돌파
```

---

## 5. 기울기 흐름 모니터링

```c
// 기울기 텐서의 L2 노름 계산 (진단용)
float grad_norm(const float *grad, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += grad[i] * grad[i];
    return sqrtf(sum);
}

// 훈련 중 기울기 노름 모니터링 (레이어별)
void print_gradient_norms(VGGNet *model) {
    printf("블록별 기울기 노름:\n");
    for (int blk = 0; blk < 5; blk++) {
        float norm = grad_norm(model->blocks[blk].conv_w[0],
                               model->blocks[blk].C_out * model->blocks[blk].C_in * 9);
        printf("  블록 %d: ||dW|| = %.6f\n", blk + 1, norm);
    }
}
// 예상 출력 (정상 훈련 시):
// 블록 5: ||dW|| = 0.002341
// 블록 4: ||dW|| = 0.001987
// 블록 3: ||dW|| = 0.000812   ← 더 작음 — 일부 소실
// 블록 2: ||dW|| = 0.000213   ← 추가 감소
// 블록 1: ||dW|| = 0.000047   ← 스킵 연결 없이 심각한 소실
```

---

## 6. VGG vs AlexNet vs ResNet 요약

```
                AlexNet     VGG-16      ResNet-50
연도            2012        2014        2015
깊이            8 레이어    16 레이어   50 레이어
파라미터        6천만       1억 3,800만 2,500만
ImageNet top-1  57.1%       74.4%       76.1%
스킵 연결       없음        없음        있음
완전연결 레이어  3개(대형)   3개(대형)   없음(GAP)

메모리 (순전파, batch=1, FP32):
  AlexNet:   ~4 MB 활성값
  VGG-16:  ~500 MB 활성값 (GPU 메모리를 압도!)
  ResNet-50: ~100 MB 활성값
```

VGG의 활성값 메모리(500MB)는 정확도가 아닌 메모리 때문에 실제 운용에서 ResNet으로 교체된 이유입니다.

---

## 핵심 정리

- **VGG 설계 원칙**: 오직 3×3 합성곱만 사용하고, 각 MaxPool에서 채널 수를 두 배로 늘림
- 두 개의 3×3 합성곱 = 하나의 5×5 수용 필드, 파라미터 28% 절감, 비선형성 추가
- **VGG의 1억 3,800만 파라미터 중 89%가 완전연결 레이어에 있음** — GAP는 이를 완전히 제거
- 깊은 네트워크(20개 레이어 이상)는 스킵 연결 없이 기울기 저하 장벽에 부딪힘 — VGG-19는 VGG-16보다 거의 개선되지 않음
- ReLU는 tanh 대비 기울기 흐름을 돕지만 깊은 네트워크 저하를 해결하지 못함 — 이를 위해서는 잔차 연결(residual connection)이 필요했음

---

**다음**: [16. ResNet과 스킵 연결](./16_ResNet_and_Skip_Connections.md) — 잔차 블록, 항등 및 투영 지름길, 스킵 연결이 기울기 소실 문제를 해결하는 이유, CIFAR-10을 위한 ResNet-20 구현.
