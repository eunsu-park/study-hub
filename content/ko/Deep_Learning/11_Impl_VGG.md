# 11. VGG

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 균일한 3×3 합성곱 필터를 사용하여 매우 깊은 네트워크를 구성하는 VGGNet의 아키텍처 철학을 설명합니다.
2. 중첩된 3×3 합성곱과 단일 대형 합성곱 간의 수용 영역(Receptive Field) 동치성을 분석하고 파라미터 수를 비교합니다.
3. VGG-16과 VGG-19의 구성(블록 구조, 채널 진행, 분류기 헤드 설계)을 설명합니다.
4. 모듈식 블록 구성을 사용하여 PyTorch에서 VGGNet 계열(예: VGG-16)을 처음부터 구현합니다.
5. 파인튜닝(Fine-tuning) 기법을 사용하여 사전 훈련된 VGGNet으로 커스텀 분류 작업에 전이 학습(Transfer Learning)을 적용합니다.
6. VGGNet의 한계(파라미터 수, 메모리 사용량)를 식별하고 이후 아키텍처들이 이를 어떻게 해결했는지 설명합니다.

## 개요

VGGNet은 2014년 ILSVRC에서 2위를 차지한 모델로, Karen Simonyan과 Andrew Zisserman이 제안했습니다. "Very Deep Convolutional Networks for Large-Scale Image Recognition" 논문에서 **3x3 작은 필터를 깊게 쌓는 것**이 효과적임을 보여주었습니다.

---

## 수학적 배경

### 이론: 파라미터와 FLOP 분석

VGG-16은 ~1억 3800만 개 파라미터를 가집니다. 어디 있을까요?

- Conv 층 (총 13개): ~1470만 params (총의 ~10%)
- FC 층 (총 3개): ~1억 2360만 params (총의 ~90%)

FC 층이 지배적인데, 첫 번째 것이 `7 * 7 * 512 = 25,088` 활성화를 4,096 유닛으로 매핑하기 때문입니다: 한 행렬에 `25,088 * 4,096 = ~1억 300만` 파라미터. 현대 아키텍처(ResNet, DenseNet, ViT)는 이를 **global average pooling** 다음 작은 FC 층 하나로 대체하여, FC 파라미터 비용을 10-100배 낮춥니다. VGG의 FC 지배성이 그것이 배포에서 인기를 잃은 가장 큰 단일 이유입니다.

FLOP의 경우 그림이 뒤집힙니다: 합성곱이 모든 공간 위치에 적용되기 때문에 지배적입니다. VGG-16 추론은 ~15.5 GFLOPs이며 대부분 conv 층에 있고, (거대한) FC 층은 한 번만 실행되어 작게 기여합니다.


### 1. 3x3 필터 스택의 효과

```
왜 3x3 필터를 여러 개 쌓는가?

2개의 3x3 conv ≈ 1개의 5x5 conv (같은 receptive field)
3개의 3x3 conv ≈ 1개의 7x7 conv

장점:
1. 파라미터 수 감소:
   - 7x7: 49C² 파라미터
   - 3x3 × 3: 27C² 파라미터 (45% 감소)

2. 비선형성 증가:
   - 7x7: 1개의 ReLU
   - 3x3 × 3: 3개의 ReLU → 더 복잡한 함수 학습 가능
```

### 2. Receptive Field 계산

```
레이어가 쌓일수록 receptive field 증가:

RF = (RF_prev - 1) × stride + kernel_size

예시 (stride=1, kernel=3):
- Layer 1: RF = 3
- Layer 2: RF = 5
- Layer 3: RF = 7
- Layer 4: RF = 9
...

MaxPool (kernel=2, stride=2) 후:
- RF가 2배로 확장
```

### 3. Feature Map 크기 변화

```
Conv (stride=1, padding=1, kernel=3):
  H_out = H_in  (크기 유지)

MaxPool (kernel=2, stride=2):
  H_out = H_in / 2  (크기 절반)

224 → [Conv×2] → 224 → Pool → 112 → [Conv×2] → 112 → Pool → 56 → ...
```

---

## VGG 아키텍처

### 이론: 깊이 절제로서의 VGG-11/13/16/19

원래 VGG 논문은 같은 전체 패턴으로 네 깊이를 비교했습니다:

| 변형 | Conv 층 | Params | ImageNet top-5 오차 |
|------|---------|--------|---------------------|
| VGG-11  | 8           | 133M   | 10.4% |
| VGG-13  | 10          | 133M   |  9.9% |
| VGG-16  | 13          | 138M   |  8.8% |
| VGG-19  | 16          | 144M   |  9.0% |

두 가지 시사점:

1. **깊이는 도움이 됩니다, 어느 한계까지.** VGG-19는 실제로 VGG-16보다 약간 나빴고, 저자들은 이를 그 깊이에서의 최적화 어려움 때문이라고 보았습니다. ResNet(다음 해)은 정확히 이를 해결했습니다: 잔차 연결이 네트워크가 152+ 층까지 최적화 붕괴 없이 가게 했습니다.
2. **대부분 파라미터가 FC 헤드에 있으므로**, 더 깊은 conv 적층이 총 파라미터 수를 거의 변경하지 않습니다. 이는 VGG를 깔끔한 실험 설계로 만들었지만 파라미터 효율 네트워크의 빈약한 템플릿으로 만들었습니다.


### 이론: VGG 원리

VGG의 아키텍처는 세 규칙을 따릅니다:

1. 모든 합성곱은 `3x3`, 스트라이드 1, 패딩 1 (출력 공간 크기 변경 없음).
2. 모든 풀링은 `2x2`, 스트라이드 2 (출력 공간 크기 절반).
3. 공간 크기가 절반될 때마다 채널 수가 두 배: `64 -> 128 -> 256 -> 512 -> 512`.

Conv 블록은 그룹으로 옵니다("블록" 구조): `Conv -> ReLU`를 2-3번 반복한 다음 `MaxPool`. VGG-16은 블록당 `[2, 2, 3, 3, 3]` 패턴의 conv를 가지며, 총 13개 conv 층 + 3개 완전 연결 층 = 16개 가중치 층입니다.

두 개 적층된 `3x3` conv는 채널당 `2 * 9 = 18`개 가중치로 `5x5` 수용 영역을 만들고; 하나의 `5x5` conv는 같은 RF를 `25`개 가중치로 만듭니다. 세 개 적층된 `3x3`은 `27` vs `49`로 `7x7` RF를 만듭니다. VGG는 커널을 더 넓게 하기보다 더 깊이 가는 것으로 *파라미터당 수용 영역*을 삽니다.


### VGG 변형 비교

| 구성 | VGG11 | VGG13 | VGG16 | VGG19 |
|------|-------|-------|-------|-------|
| Conv Layers | 8 | 10 | 13 | 16 |
| FC Layers | 3 | 3 | 3 | 3 |
| Total Layers | 11 | 13 | 16 | 19 |
| Parameters | 133M | 133M | 138M | 144M |

### VGG16 상세 구조

```
입력: 224×224×3 RGB 이미지

Block 1: [Conv3-64] × 2 + MaxPool
  (224×224×3) → (224×224×64) → (112×112×64)

Block 2: [Conv3-128] × 2 + MaxPool
  (112×112×64) → (112×112×128) → (56×56×128)

Block 3: [Conv3-256] × 3 + MaxPool
  (56×56×128) → (56×56×256) → (28×28×256)

Block 4: [Conv3-512] × 3 + MaxPool
  (28×28×256) → (28×28×512) → (14×14×512)

Block 5: [Conv3-512] × 3 + MaxPool
  (14×14×512) → (14×14×512) → (7×7×512)

Classifier:
  Flatten: 7×7×512 = 25,088
  FC1: 25088 → 4096 + ReLU + Dropout
  FC2: 4096 → 4096 + ReLU + Dropout
  FC3: 4096 → 1000 (classes)

파라미터 분포:
- Conv layers: ~15M (11%)
- FC layers: ~124M (89%)  ← 대부분!
```

### VGG 설정 (Configuration)

```python
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# 'M' = MaxPool
```

---

## 파일 구조

```
04_VGG/
├── README.md                      # 이 파일
├── pytorch_lowlevel/
│   └── vgg_lowlevel.py           # F.conv2d, F.linear 사용
├── paper/
│   └── vgg_paper.py              # 논문 아키텍처 정확 재현
└── exercises/
    ├── 01_feature_visualization.md   # 각 블록 feature map 시각화
    └── 02_transfer_learning.md       # 사전학습 가중치 활용
```

---

## 핵심 개념

### 이론: BatchNorm 없는 VGG

VGG는 BatchNorm보다 1년 앞섭니다. 그것은 다음을 사용해 성공적으로 학습되었습니다(16-19 층은 당시 매우 깊은 것으로 여겨짐):

- **신중한 초기화**: 더 얕은 네트워크를 사전학습한 다음 그 가중치를 초기화로 사용해 층을 추가.
- **수동 스케줄의 작은 학습률** (검증 손실이 평탄화되면 10배 감소).
- **긴 학습**: ImageNet에서 74 에폭, 몇 주의 계산.

BN이 도입되었을 때 VGG-BN 버전은 훨씬 쉽게 학습되었습니다 — 더 적은 웜 스타트 트릭, 더 큰 학습률, 더 빠른 수렴. 현대 재구현은 원래 VGG 논문이 그러지 않더라도 거의 항상 BN을 사용합니다. 이는 "네트워크 아키텍처가 학습되었다"가 동시대 도구상자에 의존한다는 것을 상기시킵니다; 많은 옛 논문 아키텍처는 더 나은 학습 인프라 덕분에 오늘 재현하기 더 쉽습니다.


### 1. Deep & Narrow vs Shallow & Wide

```
VGG 이전: 큰 필터 + 얕은 네트워크
  - AlexNet: 11×11, 5×5 필터
  - 적은 레이어

VGG: 작은 필터 + 깊은 네트워크
  - 오직 3×3 필터 (+ 일부 1×1)
  - 16~19 레이어

결론: 깊이가 성능에 매우 중요
```

### 2. 균일한 구조

```
VGG의 설계 원칙:

1. 모든 Conv는 3×3, stride=1, padding=1
2. 모든 MaxPool은 2×2, stride=2
3. 블록마다 채널 수 2배 증가 (64→128→256→512)
4. 간단하고 규칙적 → 이해/구현 용이
```

### 3. VGG의 한계

```
단점:
1. 파라미터 과다 (138M, ResNet-50: 25M)
2. 메모리 소비 큼 (FC 레이어)
3. 학습 느림
4. Gradient vanishing (깊어질수록)

후속 연구:
- GoogLeNet: Inception 모듈로 효율성
- ResNet: Skip connection으로 더 깊게
- MobileNet: Depthwise separable conv
```

### 4. VGG as Feature Extractor

```
VGG는 특징 추출기로 널리 사용:

1. Style Transfer
   - 콘텐츠: block4_conv2
   - 스타일: block1~5_conv1

2. Perceptual Loss
   - 픽셀 손실 대신 VGG 특징 비교

3. Object Detection
   - VGG backbone + detection head
```

---

## 구현 레벨

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)

- F.conv2d, F.max_pool2d, F.linear 사용
- nn.Conv2d, nn.Linear 미사용
- 파라미터 수동 초기화 및 관리
- 블록 단위 모듈화

### Level 3: Paper Implementation (paper/)

- 논문의 모든 설정 재현
- Batch Normalization 추가 (VGG-BN)
- 다양한 VGG 변형 지원

---

## 학습 체크리스트

- [ ] 3×3 필터 스택의 장점 이해
- [ ] Receptive field 계산 방법 숙지
- [ ] VGG16 아키텍처 암기
- [ ] 파라미터 분포 이해 (Conv vs FC)
- [ ] VGG를 feature extractor로 활용하는 방법
- [ ] VGG의 한계와 후속 모델 비교

---

## 참고 자료

- Simonyan & Zisserman (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- [torchvision VGG](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)
- [CS231n: ConvNets](https://cs231n.github.io/convolutional-networks/)
- [../03_CNN_LeNet/README.md](../03_CNN_LeNet/README.md)
