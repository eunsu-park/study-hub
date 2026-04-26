# 10. CNN (LeNet)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. LeNet-5의 역사적 의의를 설명하고, CNN 설계에 도입한 아키텍처 혁신을 서술합니다.
2. 입력 크기, 커널 크기, 패딩(Padding), 스트라이드(Stride)가 주어졌을 때 합성곱 층의 출력 공간 차원을 계산합니다.
3. LeNet-5에서 평균 풀링(Average Pooling, 서브샘플링)의 역할을 설명하고 공간 해상도를 어떻게 줄이는지 설명합니다.
4. PyTorch를 사용하여 LeNet-5를 처음부터 구현하고, 각 층의 파라미터와 활성화 함수(Activation Function)를 올바르게 지정합니다.
5. MNIST 데이터셋에서 LeNet-5를 훈련하고 정확도와 손실 지표를 사용하여 분류 성능을 평가합니다.
6. 학습된 합성곱 필터와 특성 맵(Feature Map)을 분석하여 각 층이 표현하는 것에 대한 직관을 개발합니다.

---

## 이론과 원리

LeNet-5 (LeCun et al. 1998)는 원조 합성곱 네트워크입니다. 오늘 재구현하는 것은 부분적으로 역사 수업이며 부분적으로 유용한 베이스라인입니다 — 이후 네트워크가 추가하는 모든 개념(ReLU, BN, 잔차, attention)을 이 출발점과 비교 측정할 수 있습니다. 아키텍처도 충분히 작아서 파라미터 수와 층별 활성화 형상을 손으로 유도할 수 있으며, 이는 어떤 CNN을 이해하기 위한 올바른 연습입니다.

이 섹션에서 다루는 내용:

- **A.** LeNet-5의 설계 선택과 1998년의 모습
- **B.** 서브샘플링(풀링)이 그때나 지금이나 비자명한 이유
- **C.** 합성곱 네트워크의 파라미터 카운팅 수학
- **D.** 현대 CNN에서 LeNet 선택을 대체한 것 (그리고 이유)

### A. LeNet-5 설계

LeNet-5는 `32 x 32` 회색조 숫자(MNIST)를 위해 설계되었습니다. 그 층 계획:

```
입력   32 x 32 x 1
C1     6 맵,   5x5 conv, 패딩 없음     ->  28 x 28 x 6
S2     2x2 average pool, 스트라이드 2   ->  14 x 14 x 6
C3     16 맵,  5x5 conv                ->  10 x 10 x 16
S4     2x2 average pool, 스트라이드 2   ->   5 x  5 x 16
C5     120 맵, 5x5 conv (=5x5 입력에 완전 연결)
F6     84 완전 연결 유닛
Out    10 RBF 유닛 (softmax 등가)
```

활성화는 **tanh**였습니다(ReLU 아님; ReLU는 2010까지 표준이 되지 않음). 옵티마이저는 momentum이 있는 신중하게 손수 튜닝된 SGD였습니다. 총 파라미터 수는 약 60,000개 — 가장 작은 현대 CNN보다도 3-4 자릿수 적습니다.

### B. 그때와 지금의 서브샘플링(풀링)

LeNet의 S2와 S4 층은 *학습 가능한* 서브샘플링을 했습니다: 각 `2x2` 패치가 평균된 다음, 학습 가능한 스칼라가 곱해지고 학습 가능한 편향이 더해진 후 tanh를 거쳤습니다. 현대 아키텍처는 이를 파라미터 없는 평범한 max- 또는 average-풀링으로 단순화했습니다. 실험이 학습 가능한 서브샘플링이 거의 이점이 없고 추가 파라미터 비용이 든다는 것을 보였기 때문입니다. 이 결정은 1998년에는 아무것도 절약하지 못했지만(어쨌든 네트워크가 작았음) 깊은 CNN이 모든 파라미터로 진짜 일을 해야 할 때 나중에 많이 중요해졌습니다.

더 깊은 질문 — 서브샘플링이 무엇을 해야 하는가 — 는 여러 현대적 답이 있습니다:

- **Max 풀링**: 각 패치의 가장 강한 활성화 유지(잡음에 강건).
- **Average 풀링**: 평균 유지(더 부드러움, 덜 뾰족함).
- **Strided 합성곱**: 다운샘플링 필터 학습(가장 표현력 있음, ResNet에서 사용).
- **Attention 풀링 / Global average 풀링**: 학습된 가중 방식으로 공간적 집계(현대 분류기와 ViT의 CLS 토큰에서 사용).

### C. 파라미터 카운팅

`C_in` 입력 채널, `C_out` 출력 채널, 커널 `K x K`인 합성곱층의 파라미터 수는:

```
가중치 = K * K * C_in * C_out
편향  = C_out
총    = K * K * C_in * C_out + C_out
```

LeNet의 C1: `5 * 5 * 1 * 6 + 6 = 156`. C3: `5 * 5 * 6 * 16 + 16 = 2,416`. C5: `5 * 5 * 16 * 120 + 120 = 48,120` — C5 층 단독으로 네트워크 가중치의 ~80%를 차지하는데, 그 시점에서 텐서가 공간적으로 작지만 많은 채널을 가지고 많은 출력 채널에 연결되기 때문입니다.

이는 반복되는 패턴입니다: 파라미터 수는 채널이 적은 초기 층이 아니라 많은 채널을 혼합하는 층에 의해 지배됩니다.

### D. LeNet의 선택을 대체한 것

| LeNet (1998) | 현대 대체 | 이유 |
|--------------|-----------|------|
| tanh 활성화 | ReLU / GELU | 비포화, 그래디언트 소실 회피 |
| 학습 가능 풀링 | Max / strided conv | 더 단순, 추가 파라미터 이점 없음 |
| 손 튜닝 SGD | Adam / AdamW | 파라미터별 적응 학습률 |
| 정규화 없음 | Dropout, BN, weight decay | 네트워크가 커지면서 필요해짐 |
| 5x5 커널 | 3x3 적층 | 파라미터당 더 표현력 |
| RBF 출력 | Softmax + cross-entropy | 더 잘 동작하는 그래디언트 |

이들 각각은 후속 레슨의 주제입니다. LeNet은 그것들 모두를 비교 측정하는 베이스라인입니다.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| LeNet 층 계획 | Conv2d / AvgPool / Linear의 `nn.Sequential` |
| 현대화된 활성화 | tanh 대신 `nn.ReLU()` |
| 파라미터 카운팅 | `sum(p.numel() for p in model.parameters())` |
| 학습 레시피 | `Adam` + cross-entropy + 신중한 초기화 (필요 없음) |

---


## 개요

LeNet-5는 Yann LeCun이 1998년에 제안한 최초의 성공적인 Convolutional Neural Network입니다. 손글씨 숫자 인식(MNIST)에서 뛰어난 성능을 보여주었으며, 현대 CNN의 기초가 되었습니다.

---

## 수학적 배경

### 1. Convolution 연산

```
2D Convolution:
(I * K)[i,j] = Σ_m Σ_n I[i+m, j+n] · K[m, n]

여기서:
- I: 입력 이미지 (H × W)
- K: 커널/필터 (k_h × k_w)
- *: convolution 연산

출력 크기:
H_out = (H_in + 2P - K) / S + 1
W_out = (W_in + 2P - K) / S + 1

- P: padding
- S: stride
- K: kernel size
```

### 2. Pooling 연산

```
Max Pooling:
y[i,j] = max(x[i*s:i*s+k, j*s:j*s+k])

Average Pooling:
y[i,j] = mean(x[i*s:i*s+k, j*s:j*s+k])

목적:
1. 공간 해상도 감소 (down-sampling)
2. Translation invariance 증가
3. 파라미터/계산량 감소
```

### 3. Backpropagation through Convolution

```
Forward:
Y = X * W + b

Backward:

∂L/∂W = X * ∂L/∂Y  (cross-correlation)

∂L/∂X = ∂L/∂Y * rot180(W)  (full convolution)

∂L/∂b = Σ ∂L/∂Y
```

---

## LeNet-5 아키텍처

```
입력: 32×32 흑백 이미지

Layer 1: Conv (5×5, 6 filters) → 28×28×6
         + Tanh + AvgPool (2×2) → 14×14×6

Layer 2: Conv (5×5, 16 filters) → 10×10×16
         + Tanh + AvgPool (2×2) → 5×5×16

Layer 3: Conv (5×5, 120 filters) → 1×1×120
         + Tanh

Layer 4: FC (120 → 84) + Tanh

Layer 5: FC (84 → 10) (출력)

파라미터:
- Conv1: 5×5×1×6 + 6 = 156
- Conv2: 5×5×6×16 + 16 = 2,416
- Conv3: 5×5×16×120 + 120 = 48,120
- FC1: 120×84 + 84 = 10,164
- FC2: 84×10 + 10 = 850
- 총: ~61,706 파라미터
```

---

## 파일 구조

```
03_CNN_LeNet/
├── README.md                      # 이 파일
├── numpy/
│   ├── conv_numpy.py             # NumPy로 Convolution 구현
│   ├── pooling_numpy.py          # NumPy로 Pooling 구현
│   └── lenet_numpy.py            # 전체 LeNet NumPy 구현
├── pytorch_lowlevel/
│   └── lenet_lowlevel.py         # F.conv2d 사용, nn.Conv2d 미사용
├── paper/
│   └── lenet_paper.py            # 논문 아키텍처 정확 재현
└── exercises/
    ├── 01_visualize_filters.md   # 필터 시각화
    └── 02_receptive_field.md     # 수용 영역 계산
```

---

## 핵심 개념

### 1. Local Connectivity

```
Fully Connected:
- 모든 입력이 모든 출력에 연결
- 파라미터: H_in × W_in × H_out × W_out

Convolution:
- 로컬 영역만 연결 (커널 크기)
- 파라미터: K × K × C_in × C_out
- 파라미터 공유로 효율적
```

### 2. Parameter Sharing

```
같은 필터가 이미지 전체에 적용
→ Translation equivariance
→ 어떤 위치에서든 같은 특징 감지
```

### 3. Hierarchical Features

```
Layer 1: 엣지, 코너 (저수준)
Layer 2: 텍스처, 패턴 (중수준)
Layer 3: 부분 객체 (고수준)
Layer 4+: 전체 객체 (의미론적)
```

---

## 구현 레벨

### Level 1: NumPy From-Scratch (numpy/)
- Convolution을 루프로 직접 구현
- im2col 최적화
- Backpropagation 수동 구현

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- F.conv2d, F.max_pool2d 사용
- nn.Conv2d 미사용
- 파라미터 수동 관리

### Level 3: Paper Implementation (paper/)
- 원본 논문 아키텍처 재현
- Tanh 활성화 (ReLU 대신)
- Average Pooling (Max 대신)

---

## 학습 체크리스트

- [ ] Convolution 수식 이해
- [ ] 출력 크기 계산 공식 암기
- [ ] im2col 기법 이해
- [ ] Conv backward 유도
- [ ] Max pooling backward 이해
- [ ] LeNet 아키텍처 암기

---

## 참고 자료

- LeCun et al. (1998). "Gradient-Based Learning Applied to Document Recognition"
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [CNN 기초 (Convolutional Neural Networks)](./07_CNN_Basics.md)
