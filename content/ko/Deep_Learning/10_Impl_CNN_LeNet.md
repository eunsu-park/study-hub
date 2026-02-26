# 03. CNN (LeNet)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. LeNet-5의 역사적 의의를 설명하고, CNN 설계에 도입한 아키텍처 혁신을 서술합니다.
2. 입력 크기, 커널 크기, 패딩(Padding), 스트라이드(Stride)가 주어졌을 때 합성곱 층의 출력 공간 차원을 계산합니다.
3. LeNet-5에서 평균 풀링(Average Pooling, 서브샘플링)의 역할을 설명하고 공간 해상도를 어떻게 줄이는지 설명합니다.
4. PyTorch를 사용하여 LeNet-5를 처음부터 구현하고, 각 층의 파라미터와 활성화 함수(Activation Function)를 올바르게 지정합니다.
5. MNIST 데이터셋에서 LeNet-5를 훈련하고 정확도와 손실 지표를 사용하여 분류 성능을 평가합니다.
6. 학습된 합성곱 필터와 특성 맵(Feature Map)을 분석하여 각 층이 표현하는 것에 대한 직관을 개발합니다.

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
